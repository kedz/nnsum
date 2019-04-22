import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers 


@register_module("attention.feed_forward")
class FeedForwardAttention(Module):

    hparams = hparam_registry()

    @hparams()
    def context_size(self):
        pass

    @hparams()
    def query_size(self):
        pass

    @hparams()
    def hidden_size(self):
        pass

    @hparams(default=True)
    def compute_composition(self):
        return self._compute_composition

    def init_network(self):
        self._context_network = nnsum2.layers.FullyConnected(
            in_feats=self.context_size,
            out_feats=self.hidden_size,
            activation="none",
            bias=False)
        self._query_network = nnsum2.layers.FullyConnected(
            in_feats=self.query_size,
            out_feats=self.hidden_size,
            activation="none",
            bias=True)
        self._key_network = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, context, query, context_mask=None, attention_state=None):
        # context is batch x ctx_len x hidden_size
        # query is query_len x batch x hidden_size

        context_hidden = self._context_network(context.permute(1, 0, 2))
        query_hidden = self._query_network(query)

        scores = []
        for query_hidden_step in query_hidden.split(1, dim=0):
            # score is batch x 1 x context length
            score = self._key_network(
                torch.tanh(query_hidden_step + context_hidden)
            ).permute(1, 2, 0)
            scores.append(score)
        scores = torch.cat(scores, 1)
        if context_mask is not None:
            # Transform mask from (batch_size x source_length) to
            # (batch_size x 1 x source_length). 
            context_mask = context_mask.unsqueeze(1)
            scores = scores.masked_fill_(context_mask, float("-inf"))
        attention = torch.softmax(scores, 2)

        if self.compute_composition:
            composition = attention.bmm(context).permute(1, 0, 2)
            attention = attention.permute(1, 0, 2)
            return attention, None, composition
        else:
            attention = attention.permute(1, 0, 2)
            return attention, None

    def initialize_parameters(self):
        self._context_network.initialize_parameters()
        self._query_network.initialize_parameters()
        nn.init.xavier_normal_(self._key_network.weight)
