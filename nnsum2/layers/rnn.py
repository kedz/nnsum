import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.rnn_state import RNNState


@register_module("layers.rnn")
class RNN(Module):
    
    hparams = hparam_registry()

    @hparams()
    def input_dims(self):
        pass

    @hparams()
    def rnn_dims(self):
        pass

    # Choices are lstm, gru, or rnn
    @hparams(default="lstm")
    def cell(self):
        pass

    @hparams(default=1)
    def layers(self):
        pass

    @hparams(default=0.0)
    def dropout(self):
        pass

    @hparams(default=False)
    def bidirectional(self):
        pass

    @property
    def output_dims(self):
        return self.rnn_dims * (1 + self.bidirectional)

    def init_network(self):

        if self.dropout > 0 and self.layers > 1:
            rnn_dropout = self.dropout 
        else:
            rnn_dropout = 0.
        
        if self.cell == "rnn":
            cons = nn.RNN
        elif self.cell == "gru":
            cons = nn.GRU
        else:
            cons = nn.LSTM

        self._network = cons(self.input_dims, self.rnn_dims,
                             bidirectional=self.bidirectional,
                             num_layers=self.layers,
                             dropout=rnn_dropout)

    def forward(self, inputs, lengths=None, state=None):

        if isinstance(state, RNNState):
            state = [state[0], state[1]]
        
        pack_inputs = lengths is not None and torch.any(lengths.ne(lengths[0]))

        if pack_inputs:
            if not torch.all(lengths[:-1] >= lengths[1:]):
                raise ValueError("Inputs must be sorted by decreasing length.")
            inputs = pack_padded_sequence(inputs, lengths)

        output, state = self._network(inputs, state)

        if pack_inputs:
            output, _ = pad_packed_sequence(output)

        output = F.dropout(output, p=self.dropout, training=self.training,
                           inplace=False)

        if isinstance(state, (list, tuple)):
            state = [F.dropout(state_i, p=self.dropout, training=self.training,
                               inplace=True)
                     for state_i in state]
        else:
            state = F.dropout(state, p=self.dropout, training=self.training,
                              inplace=True)

        # TODO add some container to this for outputs
        return output, RNNState.new_state(state)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)    
            else:
                nn.init.normal_(param)           
    
    def set_dropout(self, dropout):
        self._network.dropout = dropout
        self._dropout = dropout
