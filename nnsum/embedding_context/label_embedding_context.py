import torch
import torch.nn as nn

from collections import OrderedDict


class LabelEmbeddingContext(nn.Module):
    def __init__(self, vocab, embedding_size, name=None):
        super(LabelEmbeddingContext, self).__init__()
        self._vocab = vocab
        self._predictor = nn.Linear(embedding_size, len(vocab))
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def vocab(self):
        return self._vocab
  
    @property
    def named_vocabs(self):
        return OrderedDict({self.name: self.vocab})
 
    @property
    def output_size(self):
        return len(self._vocab)

    def forward(self, inputs):
        return self._predictor(inputs)
