import torch
import torch.nn as nn
import torch.nn.functional as F


class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()
        print("Pay No Attention")

    def forward(self, context, query, length):
        return query, None
