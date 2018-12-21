import torch
import numpy as np
import ujson as json
from collections import OrderedDict
from multiprocessing import Pool
import logging


def _class_weights_helper(args):
    path, sentence_limit = args
    data = json.loads(path.read_text())
    labels = data["labels"]
    if sentence_limit is not None:
        labels = labels[:sentence_limit]
    
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    return np.array([num_neg, num_pos])
    
def compute_class_weights(labels_dir, num_procs, sentence_limit=None):
    logging.info(" Computing class weights...")
    
    pool = Pool(num_procs)

    data = [(x, sentence_limit) for x in labels_dir.glob("*.json")]
    label_counts = np.array([0, 0])
    for i, result in enumerate(
            pool.imap_unordered(_class_weights_helper, data), 1):

        print("{}/{}".format(i, len(data)), 
              end="\r" if i < len(data) else "\n",
              flush=True)
        label_counts += result

def get_label_counts(dataloader, pad_index=-1):
    counts = OrderedDict()
    for name, vocab in dataloader.target_vocabs.items():
        counts[name] = torch.zeros(len(vocab))
    for batch in dataloader:
        for name, labels in batch["targets"].items():
            labels = labels.detach().numpy()
            for l, c in zip(*np.unique(labels, return_counts=True)):
                l = int(l)
                c = int(c)
                if l == pad_index:
                    continue
                counts[name][l] += c
    for cls in counts.keys():
        d = OrderedDict()
        for i, c in enumerate(counts[cls].tolist()):
            label = dataloader.target_vocabs[cls][i]
            d[label] = c
        counts[cls] = d
    return counts

def get_balanced_weights(label_counts, gpu=-1):
    named_weights = OrderedDict()
    for cls, counts in label_counts.items():
        num_labels = len(counts)
        num_samples = sum(counts.tolist())
        weights = torch.FloatTensor(num_labels)
        for i, count in enumerate(counts.tolist()):
            weights[i] = num_samples / (num_labels * count)
        if gpu > -1:
            weights = weights.cuda(gpu)
        named_weights[cls] = weights
    return named_weights
