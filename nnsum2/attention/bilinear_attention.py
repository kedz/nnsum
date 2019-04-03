import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np


@register_module("attention.bilinear_attention")
class BiLinearAttention(Module):

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

        if context_mask is not None:
            scores.data.masked_fill_(context_mask.unsqueeze(1), float("-inf"))
        
        attn = torch.softmax(scores, 2)

        if context_mask is not None:
            nan_mask = torch.all(context_mask, dim=1).view(-1, 1, 1)
            attn.data.masked_fill_(nan_mask, 0.)

        if self.compute_composition:
            composition = attn.bmm(context).permute(1, 0, 2)
            attn = attn.permute(1, 0, 2)
            return attn, None, composition
        else:
            attn = attn.permute(1, 0, 2)
            return attn, None
