import numpy as np
import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events
from nnsum.metrics import PerlRouge
#from ignite.metrics import Metric

import tempfile



import logging
import sys
import os
from collections import defaultdict
import rouge_papier





def compute_class_weights(dataset):
    logging.info(" Computing class weights...")

    labels = torch.cat([item.targets for item in dataset], 0)

    label_counts = np.array([0, 0])
    labels, counts = np.unique(
    labels.numpy(), return_counts=True)
    for label, count in zip(labels, counts):
        label_counts[label] = count
    logging.info(" Counts y=0: {}, y=1 {}".format(*label_counts))
    weight = label_counts[0] / label_counts[1]
    logging.info(" Reweighting y=1 by {}\n".format(weight))
    return weight


def label_mle_trainer(model, optimizer, train_dataloader,
                      validation_dataloader, max_epochs=10, pos_weight=None,
                      grad_clip=5):

    trainer = create_trainer(model, optimizer, pos_weight=pos_weight, 
                             grad_clip=grad_clip)

    evaluator = create_evaluator(model, validation_dataloader, 
                                 summary_length=100, 
                                 delete_temp_files=True)

    PerlRouge().attach(evaluator, "rouge")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(validation_dataloader)
        
        metrics = evaluator.state.metrics
        #print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #                          .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

        print("Rouge1={}".format(metrics["rouge1"]))
    
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

        return avg_bce.item()

    trainer = Engine(_update)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Train X-Entropy: {:.3f}".format(
            trainer.state.epoch, trainer.state.output), end="\r", flush=True)
    
    return trainer

def create_evaluator(model, dataloader, summary_length=100, 
                     delete_temp_files=True):

    def _evaluator(engine, batch):

        model.eval()
        with torch.no_grad():
            path_data = []
                
            texts = model.predict(batch, max_length=summary_length)

            for text, ref_paths in zip(texts, batch.reference_paths):
                
                summary = "\n".join(text)                

                with tempfile.NamedTemporaryFile(
                        mode="w", delete=False) as fp:
                    fp.write(summary)

                path_data.append([fp.name, [str(x) for x in ref_paths]])

        return path_data

    evaluator = Engine(_evaluator)

    return evaluator

#            config_text = rouge_papier.util.make_simple_config_text(path_data)
#            config_path = manager.create_temp_file(config_text)
#            df = rouge_papier.compute_rouge(
#                config_path, max_ngram=2, lcs=False, 
#                remove_stopwords=remove_stopwords,
#                length=summary_length)
#            print(df)
#            exit()
#            return df[-1:], hist



    return Engine(_evaluator)


def train_epoch(optimizer, model, dataloader, pos_weight=None, grad_clip=5, 
                tts=True):
    model.train()
    total_xent = 0
    total_els = 0
   
    max_iters = len(dataloader)
    
    for n_iter, batch in enumerate(dataloader, 1):
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

        total_xent += float(bce)
        total_els += total_sentences_batch

        if tts:
            sys.stdout.write(
                "train: {}/{} XENT={:0.6f}\r".format(
                    n_iter, max_iters, total_xent / total_els))
            sys.stdout.flush()
        elif n_iter % 500 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    if tts:
        sys.stdout.write("                           \r")
        sys.stdout.flush()
    else:
        print("")



    return total_xent / total_els

def validation_epoch(model, dataloader, pos_weight=None, 
                     remove_stopwords=True, summary_length=100, tts=True):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = len(dataloader)
    
    for n_iter, batch in enumerate(dataloader, 1):
        
        logits = model(batch)
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            reduction='sum')

        avg_bce = bce / float(total_sentences_batch)

        total_xent += float(bce)
        total_els += total_sentences_batch

        if tts:
            sys.stdout.write(
                "valid: {}/{} XENT={:0.6f}\r".format(
                    n_iter, max_iters, total_xent / total_els))
            sys.stdout.flush()
        elif n_iter % 500 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    if tts:
        sys.stdout.write("                           \r")
        sys.stdout.flush()
    else:
        print("")

    rouge_df, hist = compute_rouge(
        model, dataloader, remove_stopwords=remove_stopwords,
        summary_length=summary_length)
    r1, r2 = rouge_df.values[0].tolist()    
   
    avg_xent = total_xent / total_els 
    return avg_xent, r1 * 100, r2 * 100

def compute_rouge(model, dataloader, remove_stopwords=True,
                  summary_length=100):

    model.eval()

    hist = {}
    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataloader:
            texts, positions = model.predict(
                batch, return_indices=True,
                max_length=summary_length)
            for pos_b in positions:
                for p in pos_b:
                    hist[p] = hist.get(p, 0) + 1
            for b, text in enumerate(texts):
                id = batch.id[b]
                summary = "\n".join(text)                
                summary_path = manager.create_temp_file(summary)
                path_data.append([summary_path, 
                                  [str(x) for x in batch.reference_paths[b]]])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        return df[-1:], hist




