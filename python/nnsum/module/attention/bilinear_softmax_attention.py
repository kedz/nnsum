import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLinearSoftmaxAttention(nn.Module):
    def __init__(self):
        super(BiLinearSoftmaxAttention, self).__init__()

    def forward(self, context, query, length):
        raw_scores = torch.bmm(query, context.permute(0, 2, 1))
        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(2):
                raw_scores.data[b,:,l:].fill_(float("-inf"))

        #bs = length.size(0)
        #diag_mask = torch.diag(length.data.new(length.data.max()).fill_(1))
        #mask = diag_mask.unsqueeze(0).byte().repeat(bs, 1, 1)

        #raw_scores.data.masked_fill_(mask, float("-inf"))

        scores = F.softmax(raw_scores, 2)

        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(1):
                scores.data[b,l:].fill_(0)
        
        attended_context = torch.bmm(scores, context)

        output = torch.cat([query, attended_context], 2)
        return output, scores
