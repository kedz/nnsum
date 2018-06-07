import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, context, query, length, values=None):
        if values is None:
            values = context

        raw_scores = torch.bmm(query, context.permute(0, 2, 1)) / self.scale
        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(2):
                raw_scores.data[b,:,l:].fill_(float("-inf"))

        scores = F.softmax(raw_scores, 2)

        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(1):
                scores.data[b,l:].fill_(0)
        
        attended_values = torch.bmm(scores, values)

        return attended_values, scores
