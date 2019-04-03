import torch.nn as nn
from ..module import Module, register_module, hparam_registry

from collections import OrderedDict


@register_module("label_embedding_context")
class LabelEmbeddingContext(Module):

    hparams = hparam_registry()

    @hparams()
    def embedding_dims(self):
        pass

    @hparams(default=True)
    def embedding_bias(self):
        pass

    @hparams()
    def name(self):
        pass

    @hparams()
    def vocab(self):
        pass

    @hparams()
    def adaptor(self):
        pass

    @property
    def output_dims(self):
        return len(self.vocab)

    @property
    def named_vocabs(self):
        return OrderedDict({self.name: self.vocab})

    def label_frequencies(self):
        freqs = torch.LongTensor([self.vocab.count(w)
                                  for i, w in self.vocab.enumerate()])
        return OrderedDict({self.name: freqs})


    def init_network(self):
        self._network = nn.Linear(self.embedding_dims, self.output_dims,
                                  bias=self.embedding_bias)

    def forward(self, inputs):
        return self._network(self.adaptor(inputs))

    def initialize_parameters(self):
        self.adaptor.initialize_parameters()
        print(" Initializing label context: {}".format(self.name))
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 1.)    
            else:
                nn.init.normal_(param)    

    def set_dropout(self, dropout):
        if self._adaptor:
            self._adaptor.set_dropout(dropout)
