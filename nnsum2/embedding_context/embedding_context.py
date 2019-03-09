import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry

#from .vocab import Vocab

from collections import OrderedDict

@register_module("embedding_context")
class EmbeddingContext(Module):

    hparams = hparam_registry()

    @hparams()
    def name(self): 
        pass

    @hparams(default=True)
    def transpose(self):
        pass

    @hparams()
    def vocab(self):
        pass
 
    @hparams()
    def embedding_dims(self):
        pass
    
    @hparams(default=0.0)
    def embedding_dropout(self):
        pass
        
    @hparams(default=0.0)
    def input_dropout(self):
        pass

    # Options are zero and unknown
    @hparams(default="zero")
    def input_dropout_mode(self):
        pass

    @property
    def named_vocabs(self):
        return OrderedDict({self.name: self.vocab})

    def init_network(self):
        if self.input_dropout_mode == "unknown" and \
                self.vocab.unknown_index is None: 
            raise ValueError(
                "If input_dropout_mode == 'unknown', vocab must set" \
                " unknown_token")

        vsize = len(self.vocab)
        pad_index = self.vocab.pad_index
        self.embeddings = nn.Embedding(vsize, self.embedding_dims, 
                                       padding_idx=pad_index)
 
    def _apply_unknown_mode_input_dropout(self, inputs):

        if self.training and self.input_dropout > 0.:
            mask = torch.distributions.Bernoulli(
                probs=self.token_dropout).sample(sample_shape=inputs.size())
            mask = mask.byte()
            
            if str(inputs.device) != "cpu":
                mask = mask.cuda(inputs.device)

            return inputs.masked_fill(mask, self.vocab.unknown_index)
            
        else:
            return inputs
            
    def forward(self, inputs):

        if isinstance(inputs, dict):
            inputs = inputs[self.name]

        if self.input_dropout_mode == "unknown":
            inputs = self._apply_unknown_mode_input_dropout(inputs)

        if inputs.dim() == 2:
            if self.transpose:
                inputs = inputs.t()
            emb = self.embeddings(inputs)
        else:
            raise Exception("Input must be a dict or 2d tensor.")           
        
        if self.input_dropout_mode == "zero":
            emb = F.dropout2d(emb, p=self.input_dropout, 
                              training=self.training, inplace=True)
        emb = F.dropout(emb, p=self.embedding_dropout, training=self.training)

        return emb


    
#    @staticmethod
#    def from_vocab_size(vocab_size, embedding_size=50, pad=None, unknown=None,
#                        start=None, stop=None, **kwargs):
#        vocab = Vocab.from_vocab_size(vocab_size, pad=pad, unk=unknown,
#                                      start=start, stop=stop)    
#        return EmbeddingContext(vocab, embedding_size, **kwargs)
#
#    @staticmethod
#    def from_word_list(word_list, embedding_size=50, pad=None, unknown=None, 
#                       start=None, stop=None, **kwargs):
#        vocab = Vocab.from_word_list(word_list, pad=pad, unk=unknown,
#                                     start=start, stop=stop)    
#        return EmbeddingContext(vocab, embedding_size, **kwargs)

#
    def initialize_parameters(self):
        nn.init.normal_(self.embeddings.weight)
        if self.vocab.pad_index is not None:
            self.embeddings.weight[self.vocab.pad_index].data.fill_(0)

#    def parameters(self):
#        for p in self.embeddings.parameters():
#            if p.requires_grad:
#                yield p
#
#    def named_parameters(self, memo, submod_prefix):
#        for n, p in self.embeddings.named_parameters(memo, submod_prefix):
#            if p.requires_grad:
#                yield n, p
#
    def convert_index_tensor(self, tensor, drop_pad=True):
        assert tensor.dim() in [1, 2, 3]
        if tensor.dim() == 1:
            return [self.vocab[idx.item()] for idx in tensor
                    if idx != self.vocab.pad_index or not drop_pad]
        elif tensor.dim() == 2:
            return [[self.vocab[idx.item()] for idx in row
                     if idx != self.vocab.pad_index or not drop_pad]
                    for row in tensor] 
        else:
            return [self.convert_index_tensor(subten, drop_pad=drop_pad)
                    for subten in tensor]

