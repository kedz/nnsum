import tempfile
import time

import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs
from ignite.handlers import ModelCheckpoint

from nnsum.metrics import Loss, PerlRouge

from colorama import Fore, Style


def labels_mle_trainer(model, optimizer, train_dataloader,
                       validation_dataloader, max_epochs=10, pos_weight=None,
                       summary_length=100, remove_stopwords=True,
                       grad_clip=5):

    trainer = create_trainer(model, optimizer, pos_weight=pos_weight, 
                             grad_clip=grad_clip)

    evaluator = create_evaluator(model, validation_dataloader, 
                                 pos_weight=pos_weight, 
                                 summary_length=summary_length)

    xentropy = Loss(
        output_transform=lambda o: (o["total_xent"], o["total_examples"]))
    xentropy.attach(trainer, "x-entropy")
    xentropy.attach(evaluator, "x-entropy")

    rouge = PerlRouge(summary_length, remove_stopwords,
        output_transform=lambda o: o["path_data"])
    rouge.attach(evaluator, "rouge")

    @trainer.on(Events.STARTED)
    def init_history(trainer):
        trainer.state.training_history = {"x-entropy": []}
        trainer.state.validation_history = {"x-entropy": [], "rouge-1": [], 
                                            "rouge-2": []}
        trainer.state.min_valid_xent = float("inf")
        trainer.state.max_valid_rouge1 = float("-inf")
        trainer.state.max_valid_rouge2 = float("-inf")

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
 
        if valid_metrics["rouge"]["rouge-1"] > trainer.state.max_valid_rouge1:
            valid_metric_strings.append(
                Fore.GREEN + \
                "Rouge-1={:.3f}".format(valid_metrics["rouge"]["rouge-1"]) + \
                Style.RESET_ALL)
            trainer.state.max_valid_rouge1 = valid_metrics["rouge"]["rouge-1"]
        else:
            valid_metric_strings.append(
                "Rouge-1={:.3f}".format(valid_metrics["rouge"]["rouge-1"]))
        
        if valid_metrics["rouge"]["rouge-2"] > trainer.state.max_valid_rouge2:
            valid_metric_strings.append(
                Fore.GREEN + \
                "Rouge-2={:.3f}".format(valid_metrics["rouge"]["rouge-2"]) + \
                Style.RESET_ALL)
            trainer.state.max_valid_rouge2 = valid_metrics["rouge"]["rouge-2"]
        else:
            valid_metric_strings.append(
                "Rouge-2={:.3f}".format(valid_metrics["rouge"]["rouge-2"]))
        
        print("Epoch[{}] Validation {}".format(
            trainer.state.epoch,
            " ".join(valid_metric_strings))) 

        valid_history["x-entropy"].append(valid_metrics["x-entropy"])
        valid_history["rouge-1"].append(valid_metrics["rouge"]["rouge-1"])
        valid_history["rouge-2"].append(valid_metrics["rouge"]["rouge-2"])

        hrs, mins, secs = _to_hours_mins_secs(
            time.time() - trainer.state.start_time)
        print("Epoch[{}] Time Taken: {:02.0f}:{:02.0f}:{:02.0f}".format(
            trainer.state.epoch, hrs, mins, secs))
        print()

    def _score_func(trainer):
        model_idx = trainer.state.epoch - 1
        return trainer.state.validation_history["rouge-2"][model_idx]

    checkpoint = ModelCheckpoint("tmp_models", "s2s_ext", score_function=_score_func, require_empty=False, score_name="rouge-2")


    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'mymodel': model})
    trainer.run(train_dataloader, max_epochs=max_epochs)

def create_trainer(model, optimizer, pos_weight=None, grad_clip=5):

    def _update(engine, batch):
        optimizer.zero_grad()
        logits = model(
            batch, decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            reduction='sum')

        avg_bce = bce / float(total_sentences_batch)
        avg_bce.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        return {"total_xent": bce, 
                "total_examples": total_sentences_batch}

    trainer = Engine(_update)
   
    return trainer

def create_evaluator(model, dataloader, summary_length=100, pos_weight=None,
                     delete_temp_files=True):

    def _evaluator(engine, batch):

        model.eval()
        path_data = []

        with torch.no_grad():

            logits = model(
                batch, decoder_supervision=batch.targets.float())
            mask = batch.targets.gt(-1).float()
            total_sentences_batch = int(batch.num_sentences.data.sum())
            
            if pos_weight is not None:
                mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

            bce = F.binary_cross_entropy_with_logits(
                logits, batch.targets.float(),
                weight=mask, 
                reduction='sum')

            texts = model.predict(batch, max_length=summary_length)

            for text, ref_paths in zip(texts, batch.reference_paths):
                summary = "\n".join(text)                
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
                    fp.write(summary)
                path_data.append([fp.name, [str(x) for x in ref_paths]])

        return {"path_data": path_data, "total_xent": bce, 
                "total_examples": total_sentences_batch}

    return Engine(_evaluator)
