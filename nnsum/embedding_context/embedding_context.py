import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab

import argparse
from collections import OrderedDict


class EmbeddingContext(nn.Module):
    
    @staticmethod
    def update_command_line_options(parser):
        parser.add_argument(
            "--pretrained-embeddings", type=str, required=False,
            default=None)
        parser.add_argument("--top-k", type=int, required=False, default=None)
        parser.add_argument("--at-least", type=int, required=False, default=1)
        parser.add_argument(
            "--update-rule", type=str, required=False, default="update-all",
            choices=["update-all", "fix-all"],)
        parser.add_argument(
            "--filter-pretrained", action="store_true", default=False)

    @staticmethod
    def from_vocab_size(vocab_size, embedding_size=50, pad=None, unknown=None,
                        start=None, stop=None, **kwargs):
        vocab = Vocab.from_vocab_size(vocab_size, pad=pad, unk=unknown,
                                      start=start, stop=stop)    
        return EmbeddingContext(vocab, embedding_size, **kwargs)

    @staticmethod
    def from_word_list(word_list, embedding_size=50, pad=None, unknown=None, 
                       start=None, stop=None, **kwargs):
        vocab = Vocab.from_word_list(word_list, pad=pad, unk=unknown,
                                     start=start, stop=stop)    
        return EmbeddingContext(vocab, embedding_size, **kwargs)

    def __init__(self, vocab, embedding_size, 
                 token_dropout=0.0, token_dropout_mode="zero",
                 embedding_dropout=0.0, initializer=None, 
                 update_rule="update-all", name=None,
                 transpose=True):
        super(EmbeddingContext, self).__init__()

        if token_dropout_mode not in ["zero", "unknown"]:
            raise ValueError(
                "token_dropout_mode can only be 'zero' or 'unknown'.")

        if token_dropout_mode == "unknown" and vocab.unknown_index is None: 
            raise ValueError(
                "If token_dropout_mode == 'unknown', vocab must set" \
                " unknown_token")

        self.embeddings = nn.Embedding(
            len(vocab), embedding_size, padding_idx=vocab.pad_index)

        self._vocab = vocab
        self._token_dropout = token_dropout
        self._token_dropout_mode = token_dropout_mode
        self._embedding_dropout = embedding_dropout
        self._embedding_size = embedding_size
        self._update_rule = update_rule
        self._name = name
        self._transpose = transpose

        if initializer is not None:
            if (len(vocab) != initializer.size(0)) \
                    or (embedding_size != initializer.size(1)) \
                    or initializer.dim() != 2:
                raise Exception(
                    ("EmbbedingLayer: initializer has dims: ({}) "
                     " but expected ({}, {}).").format(
                         ",".join([str(s) for s in initializer.size()]),
                         len(vocab), embedding_size))
        self.initializer = initializer

        if update_rule == "fix-all":
            self.embeddings.weight.requires_grad = False

    def initialize_parameters(self):

        #print(" Initializing feature context: {}".format(self.name))
        nn.init.normal_(self.embeddings.weight)
        if self.vocab.pad_index is not None:
            self.embeddings.weight[self.vocab.pad_index].data.fill_(0)

    @property
    def name(self):
        return self._name

    @property
    def transpose(self):
        return self._transpose

    @property
    def vocab(self):
        return self._vocab
  
    @property
    def named_vocabs(self):
        return OrderedDict({self.name: self.vocab})
 
    @property
    def token_dropout(self):
        return self._token_dropout
   
    @property
    def token_dropout_mode(self):
        return self._token_dropout_mode

    @property
    def embedding_dropout(self):
        return self._embedding_dropout

    @property
    def output_size(self):
        return self._embedding_size

    @property
    def update_rule(self):
        return self._update_rule

    def _apply_unknown_mode_token_dropout(self, inputs):

        if self.training and self.token_dropout > 0.:
            
            mask = torch.distributions.Bernoulli(
                probs=self.token_dropout).sample(sample_shape=inputs.size())
            mask = mask.byte()
            
            if str(inputs.device) != "cpu":
                mask = mask.cuda(inputs.device)

            return inputs.masked_fill(mask, self._vocab.unknown_index)
            
        else:
            return inputs

    def forward(self, inputs):

        if isinstance(inputs, dict):
            tensor_inputs = inputs[self.name]
        else:
            tensor_inputs = inputs

        if self.token_dropout_mode == "unknown":
            tensor_inputs = self._apply_unknown_mode_token_dropout(
                tensor_inputs)

        if tensor_inputs.dim() == 2:
            if self.transpose:
                tensor_inputs = tensor_inputs.t()
            emb = self.embeddings(tensor_inputs)
        else:
            raise Exception("Input must be a dict or 2d tensor.")           
        
        if self.token_dropout_mode == "zero":
            emb = F.dropout2d(emb, p=self.token_dropout, 
                              training=self.training, inplace=True)
        emb = F.dropout(emb, p=self.embedding_dropout, training=self.training)

        return emb

    def parameters(self):
        for p in self.embeddings.parameters():
            if p.requires_grad:
                yield p

    def named_parameters(self, memo, submod_prefix):
        for n, p in self.embeddings.named_parameters(memo, submod_prefix):
            if p.requires_grad:
                yield n, p

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
