import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, embedding_context, hidden_dim=512, num_layers=1, 
                 rnn_cell="GRU", dropout=0.):
        super(RNNEncoder, self).__init__()

        rnn_cell = rnn_cell.upper()
        self._dropout = dropout
        assert rnn_cell in ["LSTM", "GRU", "RNN"]
        assert hidden_dim > 0
        assert num_layers > 0

        self._emb_ctx = embedding_context        
        self._rnn = getattr(nn, rnn_cell)(
            embedding_context.output_size, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.)
           
    @property
    def rnn(self):
        return self._rnn

    @property
    def embedding_context(self):
        return self._emb_ctx

    def forward(self, features, lengths):

        if not torch.all(lengths[:-1] >= lengths[1:]):
            raise ValueError("Inputs must be sorted by decreasing length.")

        emb = self._emb_ctx(features)        
        emb_packed = pack_padded_sequence(emb, lengths)
        context_packed, state = self._rnn(emb_packed)
        context, _ = pad_packed_sequence(context_packed, batch_first=True)
        context = F.dropout(context, p=self._dropout, training=self.training,
                            inplace=True)
        
        return context, state

    def initialize_parameters(self):
#        print(" Initializing encoder embedding context parameters.")
        self.embedding_context.initialize_parameters()
#        print(" Initializing encoder parameters.")
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 1.)
