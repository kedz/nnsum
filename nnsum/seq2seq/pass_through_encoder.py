import torch
import torch.nn as nn
import torch.nn.functional as F


class PassThroughEncoder(nn.Module):
    def __init__(self, embedding_context, init_state, init_state_dropout=.0):

        super(PassThroughEncoder, self).__init__()
        self.embedding_context = embedding_context
        self.init_state = init_state
        self.init_state_dropout = init_state_dropout

    def forward(self, features, lengths):
        emb = self.embedding_context(features).permute(1, 0, 2).contiguous()
        batch_size = emb.size(0)
        init_state = self.init_state.repeat(1, batch_size, 1)
        init_state = F.dropout(init_state, p=self.init_state_dropout, 
                               training=self.training)
        return emb, init_state

    def initialize_parameters(self):
        self.embedding_context.initialize_parameters()
        self.init_state.data.normal_()
