import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


_desc_message = """
    Averaging Sentence Encoder -- sentence embeddings are averaged word
    embeddings.
    """

class AveragingSentenceEncoder(nn.Module):
    def __init__(self, embedding_size, dropout=0.0):
        super(AveragingSentenceEncoder, self).__init__()
        self.dropout_ = dropout
        self.output_size_ = embedding_size

    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument(
            "--dropout", type=float, default=0.25, required=False,
            help="Drop probability applied to word embeddings.")
        return parser

    @property
    def size(self):
        return self.output_size_

    def forward(self, inputs, word_count):

        inputs_sum = inputs.sum(-2)
        word_count = word_count.float().masked_fill(
            word_count.eq(0), 1).unsqueeze(-1)

        inputs_mean = inputs_sum / word_count
        inputs_mean = F.dropout(
            inputs_mean, p=self.dropout_, training=self.training, inplace=True)

        return inputs_mean

    @property
    def needs_sorted_sentences(self):
        return False

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" AveragingSentenceEncoder initialization skipped"
                        " (no parameters).")
