import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLinearSigmoidAttention(nn.Module):
    def __init__(self, normalize=True):
        super(BiLinearSigmoidAttention, self).__init__()
        self.normalize = normalize
        #self.linear = nn.Linear(600, 600)

    def forward(self, context, query, length):
        if self.normalize:
            query_norm = torch.norm(query, p=2, dim=2, keepdim=True)
            query_norm.data.masked_fill_(query_norm.data.eq(0), 1)
            query = query / query_norm
            context_norm = torch.norm(context, p=2, dim=2, keepdim=True)
            context_norm.data.masked_fill_(context_norm.data.eq(0), 1)
            context_key = context / context_norm
        else:
            context_key = context


        raw_scores = torch.bmm(query, context_key.permute(0, 2, 1))
        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(2):
                raw_scores.data[b,:,l:].fill_(float("-inf"))

        #bs = length.size(0)
        #diag_mask = torch.diag(length.data.new(length.data.max()).fill_(1))
        #mask = diag_mask.unsqueeze(0).byte().repeat(bs, 1, 1)

        #raw_scores.data.masked_fill_(mask, float("-inf"))

        scores = F.sigmoid(raw_scores) #/ length.float().view(-1, 1, 1)
        scores_norm = torch.norm(scores, p=1, dim=2, keepdim=True)
        scores_norm.data.masked_fill_(scores_norm.data.lt(1), 1)
        scores = scores / scores_norm
        #print(scores)
        #input()

        for b, l in enumerate(length.data.tolist()):
            if l < raw_scores.size(1):
                scores.data[b,l:].fill_(0)
        
        attended_context = torch.bmm(scores, context)

        output = torch.cat([query, attended_context], 2)
        return output, scores
