import torch
import torch.nn as nn
import numpy as np


class DotAttention(nn.Module):
    def __init__(self, context_size, query_size, temp="auto",
                 compute_composition=True):
        super(DotAttention, self).__init__()
        if context_size != query_size:
            raise ValueError("context_size must equal query_size")
        if temp != "auto":
            if not isinstance(temp, (float, int)) or temp < 0:
                raise ValueError(
                    "temp must be non-negative float or int, or 'auto'")

        self._context_size = context_size
        self._query_size = query_size
        self._temp = temp
        self._compute_composition = compute_composition

    @property
    def context_size(self):
        return self._context_size

    @property
    def query_size(self):
        return self._query_size

    @property
    def temp(self):
        if self._temp == "auto":
            return np.sqrt(self.query_size) 
        else:
            return self._temp
    
    @property
    def compute_composition(self):
        return self._compute_composition

    def forward(self, context, query, context_mask=None):
        # context is batch x ctx_len x hidden_size
        # query is query_len x batch x hidden_size

        perm_query = query.permute(1, 2, 0)
        # scores is batch size x query length x context length
        scores = context.bmm(perm_query).permute(0, 2, 1) / self.temp

        if context_mask is not None:
            scores.data.masked_fill_(context_mask.unsqueeze(1), float("-inf"))
        
        attn = torch.softmax(scores, 2)
        if self.compute_composition:
            composition = attn.bmm(context).permute(1, 0, 2)
            attn = attn.permute(1, 0, 2)
            return attn, composition
        else:
            attn = attn.permute(1, 0, 2)
            return attn
