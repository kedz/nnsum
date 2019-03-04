import torch
import torch.nn as nn

from collections import OrderedDict


class LabelEmbeddingContext(nn.Module):
    def __init__(self, vocab, embedding_size, name=None):
        super(LabelEmbeddingContext, self).__init__()
        self._vocab = vocab
        self._predictor = nn.Linear(embedding_size, len(vocab))
        self._name = name
        self._embedding_size = embedding_size

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def name(self):
        return self._name

    @property
    def vocab(self):
        return self._vocab
  
    @property
    def named_vocabs(self):
        return OrderedDict({self.name: self.vocab})

    def label_frequencies(self):
        freqs = torch.LongTensor([self.vocab.count(w)
                                  for i, w in self.vocab.enumerate()])
        return OrderedDict({self.name: freqs})

    @property
    def output_size(self):
        return len(self._vocab)

    def forward(self, inputs):
        return self._predictor(inputs)

    def initialize_parameters(self):
        print(" Initializing label context: {}".format(self.name))
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 1.)    
            else:
                nn.init.normal_(param)    
