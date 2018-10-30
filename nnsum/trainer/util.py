import torch
import numpy as np
import ujson as json
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

#    for item in dataset:
#        labels, counts = np.unique(
#            item["targets"].numpy(), return_counts=True)
#        for label, count in zip(labels, counts):
#            label_counts[label] += count
    logging.info(" Counts y=0: {}, y=1 {}".format(*label_counts))
    weight = label_counts[0] / label_counts[1]
    logging.info(" Reweighting y=1 by {}\n".format(weight))
    return weight
