import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs
from ignite.handlers import ModelCheckpoint

from nnsum.metrics import Loss
from nnsum.sequence_cross_entropy import sequence_cross_entropy

from colorama import Fore, Style
import ujson as json

import time



def seq2seq_mle_trainer(model, optimizer, train_dataloader,
                        validation_dataloader, max_epochs=10,
                        grad_clip=5, gpu=-1, model_path=None,
                        results_path=None):
                        #, teacher_forcing=-1,
                        #create_trainer_fn=None):

    trainer = create_trainer(model, optimizer, grad_clip=grad_clip, gpu=gpu)
    evaluator = create_evaluator(model, validation_dataloader, gpu=gpu)

    xentropy = Loss(
        output_transform=lambda o: (o["total_xent"], o["total_tokens"]))
    xentropy.attach(trainer, "x-entropy")
    xentropy.attach(evaluator, "x-entropy")

    @trainer.on(Events.STARTED)
    def init_history(trainer):
        trainer.state.training_history = {"x-entropy": []}
        trainer.state.validation_history = {"x-entropy": []}
        trainer.state.min_valid_xent = float("inf")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start_time(trainer):
        trainer.state.start_time = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):

        xent = xentropy.compute()
        iterate = trainer.state.iteration
        msg = "Epoch[{}] Training {} / {}  X-Entropy: {:.3f}".format(
            trainer.state.epoch, iterate, len(train_dataloader),
            xent)
        if iterate < len(train_dataloader):
            print(msg, end="\r", flush=True)
        else:
            print(" " * len(msg), end="\r", flush=True)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_validation_loss(evaluator):

        xent = xentropy.compute()
        iterate = evaluator.state.iteration
        msg = "Epoch[{}] Validating {} / {}  X-Entropy: {:.3f}".format(
            trainer.state.epoch, iterate, len(validation_dataloader),
            xent)
        if iterate < len(validation_dataloader):
            print(msg, end="\r", flush=True)
        else:
            print(" " * len(msg), end="\r", flush=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        trainer.state.iteration = 0
        train_metrics = trainer.state.metrics
        print("Epoch[{}] Training X-Entropy={:.3f}".format(
            trainer.state.epoch, train_metrics["x-entropy"]))
        trainer.state.training_history["x-entropy"].append(
            train_metrics["x-entropy"])

        evaluator.run(validation_dataloader)
        metrics = evaluator.state.metrics
        
        valid_metrics = evaluator.state.metrics
        valid_history = trainer.state.validation_history
        valid_metric_strings = []

        if valid_metrics["x-entropy"] < trainer.state.min_valid_xent:
            valid_metric_strings.append(
                Fore.GREEN + \
                "X-Entropy={:.3f}".format(valid_metrics["x-entropy"]) + \
                Style.RESET_ALL)
            trainer.state.min_valid_xent = valid_metrics["x-entropy"]
        else:
            valid_metric_strings.append(
                "X-Entropy={:.3f}".format(valid_metrics["x-entropy"]))
 
#        if valid_metrics["rouge"]["rouge-1"] > trainer.state.max_valid_rouge1:
#            valid_metric_strings.append(
#                Fore.GREEN + \
#                "Rouge-1={:.3f}".format(valid_metrics["rouge"]["rouge-1"]) + \
#                Style.RESET_ALL)
#            trainer.state.max_valid_rouge1 = valid_metrics["rouge"]["rouge-1"]
#        else:
#            valid_metric_strings.append(
#                "Rouge-1={:.3f}".format(valid_metrics["rouge"]["rouge-1"]))
#        
#        if valid_metrics["rouge"]["rouge-2"] > trainer.state.max_valid_rouge2:
#            valid_metric_strings.append(
#                Fore.GREEN + \
#                "Rouge-2={:.3f}".format(valid_metrics["rouge"]["rouge-2"]) + \
#                Style.RESET_ALL)
#            trainer.state.max_valid_rouge2 = valid_metrics["rouge"]["rouge-2"]
#        else:
#            valid_metric_strings.append(
#                "Rouge-2={:.3f}".format(valid_metrics["rouge"]["rouge-2"]))
#        
        print("Epoch[{}] Validation {}".format(
            trainer.state.epoch,
            " ".join(valid_metric_strings))) 

        valid_history["x-entropy"].append(valid_metrics["x-entropy"])
        #valid_history["rouge-1"].append(valid_metrics["rouge"]["rouge-1"])
        #valid_history["rouge-2"].append(valid_metrics["rouge"]["rouge-2"])

        hrs, mins, secs = _to_hours_mins_secs(
            time.time() - trainer.state.start_time)
        print("Epoch[{}] Time Taken: {:02.0f}:{:02.0f}:{:02.0f}".format(
            trainer.state.epoch, hrs, mins, secs))

        print()

        if results_path:
            if not results_path.parent.exists():
                results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                json.dumps({"training": trainer.state.training_history,
                            "validation": trainer.state.validation_history}))

    if model_path:
        checkpoint = create_checkpoint(model_path)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint, {"model": model})

    trainer.run(train_dataloader, max_epochs=max_epochs)

def create_trainer(model, optimizer, grad_clip=5, gpu=-1):

    pad_index = 0

    def _update(engine, batch):
        if gpu > -1: 
            _seq2seq2gpu(batch, gpu)
        model.train()
        optimizer.zero_grad()
        logits = model(batch)
        
        tgts = batch["target_output_features"]["tokens"].t()
        total_tokens = batch["target_lengths"].sum().item()

        tot_xent = sequence_cross_entropy(logits, tgts, pad_index=pad_index)
        avg_xent = tot_xent / total_tokens
        avg_xent.backward()
        
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        return {"total_xent": tot_xent, 
                "total_tokens": total_tokens}

    trainer = Engine(_update)
   
    return trainer

def create_evaluator(model, dataloader, gpu=-1):

    pad_index = 0

    def _evaluator(engine, batch):

        if gpu > -1: 
            _seq2seq2gpu(batch, gpu)

        model.eval()

        with torch.no_grad():

            logits = model(batch)
            tgts = batch["target_output_features"]["tokens"].t()
            total_tokens = batch["target_lengths"].sum().item()

            tot_xent = sequence_cross_entropy(logits, tgts, 
                                              pad_index=pad_index)

        return {"total_xent": tot_xent, 
                "total_tokens": total_tokens}

    return Engine(_evaluator)

def create_checkpoint(model_path, metric_name="x-entropy"):
    dirname = str(model_path.parent)
    prefix = str(model_path.name)

    def _score_func(trainer):
        model_idx = trainer.state.epoch - 1
        return -trainer.state.validation_history[metric_name][model_idx]

    checkpoint = ModelCheckpoint(dirname, prefix, score_function=_score_func,
                                 require_empty=False, score_name=metric_name)
    return checkpoint

def _seq2seq2gpu(batch, gpu):
    sf = batch["source_features"]
    for feat in sf:
        sf[feat] = sf[feat].cuda(gpu)
    batch["source_lengths"] = batch["source_lengths"].cuda(gpu)
    batch["target_lengths"] = batch["target_lengths"].cuda(gpu)
    tif = batch["target_input_features"]
    for feat in tif:
        tif[feat] = tif[feat].cuda(gpu)
    tof = batch["target_output_features"]
    for feat in tof:
        tof[feat] = tof[feat].cuda(gpu)
