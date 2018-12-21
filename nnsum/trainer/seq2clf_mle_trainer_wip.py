import torch
import torch.nn.functional as F
import numpy as np

from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs
from ignite.handlers import ModelCheckpoint

from nnsum.metrics import Loss, ClassificationMetrics
from nnsum.sequence_cross_entropy import sequence_cross_entropy
from nnsum.loss import binary_entropy
import nnsum.loss

from colorama import Fore, Style
import ujson as json

import time
from collections import OrderedDict


def seq2clf_mle_trainer(model, optimizer, train_dataloader,
                        validation_dataloader, max_epochs=10,
                        grad_clip=5, gpu=-1, model_path=None,
                        results_path=None, label_weights=None,
                        max_entropy_for_missing_data=False,
                        min_attention_entropy=False,
                        use_njsd_loss=False):

                        #, teacher_forcing=-1,
                        #create_trainer_fn=None):

    trainer = create_trainer(
        model, optimizer, grad_clip=grad_clip, gpu=gpu,
        label_weights=label_weights,
        max_entropy_for_missing_data=max_entropy_for_missing_data,
        min_attention_entropy=min_attention_entropy,
        use_njsd_loss=use_njsd_loss)
    evaluator = create_evaluator(
        model, validation_dataloader, gpu=gpu,
        max_entropy_for_missing_data=max_entropy_for_missing_data,
        min_attention_entropy=min_attention_entropy,
        use_njsd_loss=use_njsd_loss)

    xentropy = Loss(
        output_transform=lambda o: (o["total_xent"], o["total_tokens"]))
    xentropy.attach(trainer, "x-entropy")
    xentropy.attach(evaluator, "x-entropy")

    if use_njsd_loss:
        njsd_loss = Loss(
            zero_ok=True,
            output_transform=lambda o: (o["njsd"]["penalty"], 
                                        o["njsd"]["active"]))
        njsd_loss.attach(trainer, "njsd")
        njsd_loss.attach(evaluator, "njsd")

    clf_metrics = OrderedDict()
    def getter(name):
        def f(out):
            return (out["labels"][name]["true"], out["labels"][name]["pred"])
        return f
    for name, vocab in model.target_embedding_context.named_vocabs.items():
        clf_metrics[name] = ClassificationMetrics(
            vocab, output_transform=getter(name))
        clf_metrics[name].attach(evaluator, name)

    if max_entropy_for_missing_data:
        ent_metrics = OrderedDict()
        def ent_getter(name):
            def f(out):
                return (out["labels"][name]["total_entropy"], 
                        out["labels"][name]["total_uncertain"])
            return f
        for name in model.target_embedding_context.named_vocabs.keys():
            ent_metrics["ent" + name] = Loss(output_transform=ent_getter(name),
                                             zero_ok=True)
            ent_metrics["ent" + name].attach(evaluator, "ent" + name)
    
#    if min_attention_entropy:
    attn_ent = Loss(
        zero_ok=True, 
        output_transform=lambda o: (o["attention_entropy"], o["total_tokens"]))
    attn_ent.attach(evaluator, "attnent")

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
        if use_njsd_loss:
            njsd = njsd_loss.compute()
            iterate = trainer.state.iteration
            msg = "Epoch[{}] Training {} / {}  X-Entropy: {:.3f} NJSD: {:.3f}".format(
                trainer.state.epoch, iterate, len(train_dataloader),
                xent, njsd)
        else:
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
        if use_njsd_loss:
            valid_metric_strings.append(
                "  NJSD={:.3f}".format(valid_metrics["njsd"]))

        #if min_attention_entropy:
        valid_metric_strings.append(
            " ATTN ENT={:6.3f}".format(valid_metrics["attnent"]))

        valid_clf_metric_strings = ["F1  "]
        for name in clf_metrics:
            valid_clf_metric_strings.append(
                "{}: {:0.3f} ".format(
                    name,
                    valid_metrics[name]["f-measure"]["macro avg."]))
            
 
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
        print(" ".join(valid_clf_metric_strings))

        valid_clf_metric_strings = ["ACC "]
        for name in clf_metrics:
            valid_clf_metric_strings.append(
                "{}: {:0.3f} ".format(
                    name,
                    valid_metrics[name]["accuracy"]))
        print(" ".join(valid_clf_metric_strings))
 

        if max_entropy_for_missing_data:
            valid_clf_metric_strings = ["ENT "]
            for name in ent_metrics:
                valid_clf_metric_strings.append(
                    "{}: {:3.0f}% ".format(
                        name[3:],
                        valid_metrics[name] * 100))
            print(" ".join(valid_clf_metric_strings))
     


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

def njsd_penalty(probs):
    prob_masks = [p.gt(.5) for p in probs]
    active_counts = sum([m.float() for m in prob_masks])
    inactive = active_counts.eq(0)
    weights = 1 / active_counts.masked_fill(active_counts.eq(0), 1.)

    sum_probs = weights * sum([p.masked_fill(~m, 0) 
                               for p, m in zip(probs, prob_masks)])
    jnt_entropy = binary_entropy(
        sum_probs, reduction="none").masked_fill(inactive, 0)

    ind_entropies = []
    for p, m in zip(probs, prob_masks):
        ind_entropy = binary_entropy(p, reduction="none").masked_fill(~m, 0)
        ind_entropies.append(ind_entropy)
    ind_entropies = weights * sum(ind_entropies)

    jsd = jnt_entropy - ind_entropies

    counts = jsd.ne(0).float().sum()
    njsd = jsd.sum() * -1.
    return njsd, counts


def create_trainer(model, optimizer, grad_clip=5, gpu=-1, label_weights=None,
                   max_entropy_for_missing_data=False,
                   min_attention_entropy=False,
                   use_njsd_loss=False):
                   
    pad_index = -1

    def _update(engine, batch):
        if gpu > -1: 
            _seq2clf2gpu(batch, gpu)
        model.train()
        optimizer.zero_grad()

        logits, attns = model(batch)

        total_xents = 0
        total_nents = 0

        for cls, cls_logits in logits.items():

            if max_entropy_for_missing_data:
                entropy_mask = batch["targets"][cls].ne(pad_index)
                class_entropy = nnsum.loss.entropy(
                    cls_logits, reduction="none")
                class_entropy.data.masked_fill_(entropy_mask, 0.)
                total_nents = total_nents \
                    - class_entropy.sum() \
                    / max((~entropy_mask).float().sum().item(), 1.)

            if label_weights is not None:
                weight = label_weights[cls]
            else:
                weight = None

            cls_avg_xent = F.cross_entropy(cls_logits, batch["targets"][cls],
                                           ignore_index=pad_index, 
                                           weight=weight)
            total_xents = total_xents + cls_avg_xent

        obj = total_xents / len(logits)
        if max_entropy_for_missing_data:
            avg_nents = total_nents / len(logits)
            obj = obj + .01 * avg_nents
 
        if min_attention_entropy:
            attn_entropy = sum([binary_entropy(a, reduction="sum") 
                                for a in attns]) 
            obj = 1. * obj + 25. * attn_entropy

        if use_njsd_loss:
            njsd, total_njsd_active = njsd_penalty(attns)
            obj = obj + 10. * njsd

        obj.backward()
          
        for v in logits.values():
            batch_size = v.size(0)
            break
        
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        result = {"total_xent": total_xents,
                  "total_tokens": batch_size}
        if use_njsd_loss:
            result["njsd"] = {"penalty": njsd, "active": total_njsd_active}
        return result

    trainer = Engine(_update)
   
    return trainer

def create_evaluator(model, dataloader, gpu=-1, min_attention_entropy=False,
                     max_entropy_for_missing_data=False, use_njsd_loss=False):

    pad_index = -1

    def _evaluator(engine, batch):

        if gpu > -1: 
            _seq2clf2gpu(batch, gpu)

        model.eval()

        with torch.no_grad():

            logits, attns = model(batch)
            total_xents = 0

            output_labels = {}
            output_entropy = {}
            for cls, cls_logits in logits.items():
               
                 
                cls_avg_xent = F.cross_entropy(
                    cls_logits, batch["targets"][cls], ignore_index=pad_index)
                total_xents = total_xents + cls_avg_xent
                output_labels[cls] = {
                    "pred": cls_logits.max(1)[1].cpu(),
                    "true": batch["targets"][cls].cpu()}

                if max_entropy_for_missing_data:
                    entropy_mask = batch["targets"][cls].ne(pad_index)
                    class_entropy = nnsum.loss.entropy(
                        cls_logits, reduction="none")
                    class_entropy.data.masked_fill_(entropy_mask, 0.)
                    class_entropy_per = class_entropy / np.log(cls_logits.size(1))
                    total_entropy_per = class_entropy_per.sum()
                    num_missing = (~entropy_mask).sum().item()

                    output_labels[cls]["total_entropy"] = total_entropy_per
                    output_labels[cls]["total_uncertain"] = num_missing
        


            for v in logits.values():
                batch_size = v.size(0)
                break
            result = {"total_xent": total_xents, 
                      "total_tokens": batch_size,
                      "labels": output_labels}
            #if min_attention_entropy:
            attn_entropy = sum([binary_entropy(a, reduction="sum") 
                                for a in attns]) 
            result["attention_entropy"] = attn_entropy
            if use_njsd_loss:
                njsd, total_njsd_active = njsd_penalty(attns)
                result["njsd"] = {"penalty": njsd, 
                                  "active": total_njsd_active}

            return result 
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

def _seq2clf2gpu(batch, gpu):
    sf = batch["source_features"]
    for feat in sf:
        sf[feat] = sf[feat].cuda(gpu)
    batch["source_lengths"] = batch["source_lengths"].cuda(gpu)
    for feat in batch["targets"]:
        batch["targets"][feat] = batch["targets"][feat].cuda(gpu)
    batch["source_mask"] = batch["source_mask"].cuda(gpu)
