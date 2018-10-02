import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads=3, head_size=25):
        super(MultiHeadAttention, self).__init__()

        input_heads = []
        self.attention = ScaledDotProductAttention(scale=np.sqrt(head_size))
        for i in range(num_heads):
            input_heads.append(
                nn.ModuleList([nn.Linear(input_size, head_size),
                               nn.Linear(input_size, head_size),
                               nn.Linear(input_size, head_size)]))
        self.input_heads = nn.ModuleList(input_heads)        
        self.linear = nn.Linear(head_size * num_heads, input_size)

    def forward(self, context, query, values, length):

        all_reads = []
        all_scores = []

        for linear_ops in self.input_heads:
            context_w = linear_ops[0](context)
            query_w = linear_ops[1](query)
            values_w = linear_ops[2](values)
            reads, scores = self.attention(
                context_w, query_w, length, values=values_w)
            all_reads.append(reads)
            all_scores.append(scores)
        all_reads = torch.cat(all_reads, 2)
        output = self.linear(all_reads)
        return output, all_scores
