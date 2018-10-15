import torch
import numpy as np

import logging

def compute_class_weights(dataset):
    logging.info(" Computing class weights...")
    label_counts = np.array([0, 0])
    for item in dataset:
        labels, counts = np.unique(
            item["targets"].numpy(), return_counts=True)
        for label, count in zip(labels, counts):
            label_counts[label] += count
    logging.info(" Counts y=0: {}, y=1 {}".format(*label_counts))
    weight = label_counts[0] / label_counts[1]
    logging.info(" Reweighting y=1 by {}\n".format(weight))
    return weight
