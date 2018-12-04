import torch
import torch.nn as nn


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()
        from warning import warn
        warn("Fix masking for dot attention.")

    def forward(self, context, query):
        # context is batch x ctx_len x hidden_size
        # query is query_len x batch x hidden_size

        perm_query = query.permute(1, 2, 0)
        scores = context.bmm(perm_query)
        attn = torch.softmax(scores.permute(0,2,1), 2)
        read = attn.bmm(context)
        output = torch.cat(
            [read.permute(1, 0, 2), query], 2)
        return output, attn.permute(1, 0, 2)

