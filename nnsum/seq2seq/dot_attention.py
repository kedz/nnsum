import torch
import torch.nn as nn


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, context, query, mask=None):
        # context is batch x ctx_len x hidden_size
        # query is query_len x batch x hidden_size

        perm_query = query.permute(1, 2, 0)
        # scores is batch size x query length x context length
        scores = context.bmm(perm_query).permute(0, 2, 1)

        if mask is not None:
            scores.data.masked_fill_(mask.unsqueeze(1), float("-inf"))
        
        attn = torch.softmax(scores, 2)
        read = attn.bmm(context)
        output = torch.cat(
            [read.permute(1, 0, 2), query], 2)
        attn = attn.permute(1, 0, 2)

        # output is query length x batch size x hidden size
        # attn is query length x batch size x context length 
        return output, attn
