import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np


@register_module("attention.accumulating_bilinear_attention")
class AccumulatingBiLinearAttention(Module):

    hparams = hparam_registry()

    @hparams()
    def context_size(self):
        return self._context_size

    @hparams()
    def query_size(self):
        return self._query_size

    @hparams(default=1.0)
    def temp(self):
        pass
        
    @property
    def real_temp(self):
        if self.temp == "auto":
            return np.sqrt(self.query_size) 
        else:
            return self.temp
    
    @hparams(default=True)
    def compute_composition(self):
        return self._compute_composition

    def init_network(self):
        self._weights = nn.Parameter(
            torch.FloatTensor(self.context_size, self.query_size).normal_())

    def forward(self, context, query, context_mask=None, attention_state=None):
        # context is batch x ctx_len x hidden_size
        # query is query_len x batch x hidden_size

        perm_query = query.permute(1, 2, 0)
        # scores is batch size x query length x context length

        batch_size, ctx_steps, ctx_dims = context.size()
        context_dot_weights = context.contiguous()\
            .view(-1, ctx_dims).mm(self._weights)\
            .view(batch_size, ctx_steps, -1)

        scores = context_dot_weights.bmm(perm_query).permute(0, 2, 1) 
        if self.real_temp != 1.:
            scores = scores / self.real_temp

        scores = scores.split(1, 1)
        
        accum = scores[0].new(scores[0].size()).fill_(0)
        attention = []
        
        if context_mask is not None:
            # Transform mask from (batch_size x source_length) to
            # (batch_size x 1 x source_length). 
            context_mask = context_mask.unsqueeze(1)

        for score in scores:
            adjusted_score = score - accum
            adjusted_score.data.masked_fill_(context_mask, float("-inf"))
            a = torch.softmax(adjusted_score, 2)
            
            #if context_mask is not None:
            #    nan_mask = torch.all(context_mask, dim=1).view(-1, 1, 1)
            #    attn.data.masked_fill_(nan_mask, 0.)

            attention.append(a)
            accum = accum + a

        attention = torch.cat(attention, 1)

        if self.compute_composition:
            composition = attention.bmm(context).permute(1, 0, 2)
            attention = attention.permute(1, 0, 2)
            return attention, accum, composition
        else:
            attention = attention.permute(1, 0, 2)
            return attention, accum
