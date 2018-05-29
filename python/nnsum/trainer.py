import numpy as np
import torch.nn.functional as F

import logging
import sys
import os
from collections import defaultdict
import rouge_papier


def compute_class_weights(dataset):
    logging.info(" Computing class weights...")
    label_counts = np.array([0, 0])
    labels, counts = np.unique(
    dataset.targets.numpy(), return_counts=True)
    for label, count in zip(labels, counts):
        label_counts[label] = count
    logging.info(" Counts y=0: {}, y=1 {}".format(*label_counts))
    weight = label_counts[0] / label_counts[1]
    logging.info(" Reweighting y=1 by {}\n".format(weight))
    return weight


def train_epoch(optimizer, model, dataset, pos_weight=None, grad_clip=5, 
                tts=True):
    model.train()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        optimizer.zero_grad()
        
        logits = model(batch.inputs, decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = batch.inputs.num_sentences.data.sum()
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_bce = bce / total_sentences_batch
        avg_bce.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        total_xent += bce.data[0]
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

def validation_epoch(model, dataset, reference_dir, pos_weight=None, 
                     remove_stopwords=True, summary_length=100, tts=True):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        
        logits = model(batch.inputs)
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = batch.inputs.num_sentences.data.sum()
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_bce = bce / total_sentences_batch

        total_xent += bce.data[0]
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
        model, dataset, reference_dir, remove_stopwords=remove_stopwords,
        summary_length=summary_length)
    r1, r2 = rouge_df.values[0].tolist()    
   
    avg_xent = total_xent / total_els 
    return avg_xent, r1 * 100, r2 * 100

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir, remove_stopwords=True,
                  summary_length=100):

    model.eval()

    hist = {}
    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            texts, positions = model.predict(
                batch.inputs, batch.metadata, return_indices=True,
                max_length=summary_length)
            for pos_b in positions:
                for p in pos_b:
                    hist[p] = hist.get(p, 0) + 1
            for b, text in enumerate(texts):
                id = batch.metadata.id[b]
                summary = "\n".join(text)                
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        return df[-1:], hist




