import torch.nn as nn


class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()

    def forward(self, context, query, mask=None):
        return query, None
