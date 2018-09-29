import numpy as np
import torch.nn.functional as F
import torch

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

def compute_loss(out, batch, mask, raml, mrt):
   
   if mrt:
     return out

   # in case of raml we need to explode the loss
   # for all the label_scores
   if raml:
     label_scores = batch.metadata.label_scores
     batch_size = len(label_scores)
     sample_size = len(label_scores[0])
     seq_size = batch.targets.size(1)

     loss_list = []
     totals = [0.0]*batch_size
     for i in range(sample_size):
       targets = batch.targets.new(batch.targets).float().fill_(0)
       for b in range(batch_size):
         #print("DEBUG: i %d, b %d, label_scores (%d,%d)" % (i, b, len(label_scores), len(label_scores[b])))
         labels = label_scores[b][i]["labels"]
         labels = labels + [0]*(seq_size-len(labels))
         assert(seq_size==len(labels))
         targets[b] = batch.targets.new(labels).float()
         totals[b] += label_scores[b][i]["score"]
       l = F.binary_cross_entropy_with_logits(
            out, targets,
            weight=mask,
            reduction='none')
       loss_list.append(l)

     # weight the losses and sum to get one number to perform
     # gradient decent on
     for i in range(sample_size):
       for b in range(batch_size):
         score = label_scores[b][i]["score"]
         loss_list[i][b] *= score / totals[b]
     
     return sum(loss_list).sum(1).sum(0)
   else:
     return F.binary_cross_entropy_with_logits(
            out, batch.targets.float(),
            weight=mask,
            reduction='sum')

def train_epoch(optimizer, model, dataset, pos_weight=None, grad_clip=5, 
                tts=True, mrt=False, raml=False):
    model.train()
    total_xent = 0
    total_els = 0
   
    #print("DEBUG: raml is set to %s" % raml)
 
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        optimizer.zero_grad()
        out = model(batch.inputs, batch.metadata) if mrt else model(batch.inputs, 
                                           decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.inputs.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        loss = compute_loss(out, batch, mask, raml=raml, mrt=mrt)

        avg_loss = loss if mrt else loss / float(total_sentences_batch)
        avg_loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        total_xent += float(loss)
        total_els += 1 if mrt else total_sentences_batch

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
                     remove_stopwords=True, summary_length=100, 
                     tts=True, mrt=False, alt_rouge=None, raml=False):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        
        out = model(batch.inputs, batch.metadata) if mrt else model(batch.inputs)
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = int(batch.inputs.num_sentences.data.sum())
        
        if pos_weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), pos_weight)

        loss = compute_loss(out, batch, mask, raml=raml, mrt=mrt)

        avg_loss = loss if mrt else loss / float(total_sentences_batch)

        total_xent += float(loss)
        total_els += 1 if mrt else total_sentences_batch

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

    rouge_df, hist, alt_score = compute_rouge(
        model, dataset, reference_dir, remove_stopwords=remove_stopwords,
        summary_length=summary_length, alt_rouge=alt_rouge)
    r1, r2 = rouge_df.values[0].tolist()    
   
    avg_xent = total_xent / total_els 
    return avg_xent, (alt_score if alt_rouge else r1) * 100, r2 * 100

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir, remove_stopwords=True,
                  summary_length=100, alt_rouge = None):

    alt_score = 0.0

    if alt_rouge:
      (ref_dicts, scorer) = alt_rouge

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

                if alt_rouge:
                  for sen, pos in zip(text, positions[b]):
                    scorer.update(sen, "%s-%s" % (id,pos))
                    #print("DEBUG: %s <===============> %s" % (sen,scorer.cache["%s-%s" % (id,pos)]))
                  alt_score += scorer.compute(ref_dicts[id])
                  
        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        return df[-1:], hist, alt_score / len(path_data)




