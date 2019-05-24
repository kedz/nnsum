from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.rnn_state import RNNState
import torch
import torch.nn as nn
import torch.nn.functional as F


@register_module("seq2seq.rnn_chasm")
class RNNChasm(Module):

    hparams = hparam_registry()

    @hparams()
    def num_layers(self):
        pass

    @hparams()
    def lstm_cell(self):
        pass
       
    @hparams(default=0.0)
    def dropout(self):
        pass

    @hparams()
    def decoder_dims(self):
        pass

    def init_network(self):

        self._output_weight = nn.Parameter(
            torch.FloatTensor(
                self.num_layers, 1, self.decoder_dims).normal_(0)
        )
        
        if self.lstm_cell:
            self._hidden_weight = nn.Parameter(
                torch.FloatTensor(self.num_layers, 1, 
                                  self.decoder_dims).normal_(0)
            )
        else:
            self._hidden_weight = None

    def forward(self, inputs):
        
        if self.lstm_cell:
            return self._lstm_forward(inputs)
        else: 
            return self._state_forward(inputs, self._output_weight)

    def _lstm_forward(self, inputs):
        if not isinstance(inputs, (list, tuple, RNNState)) and len(inputs) == 2:   
            raise Exception("Expecting lstm state!")
        output_state = self._state_forward(
            inputs[0], self._output_weight)
        hidden_state = self._state_forward(
            inputs[1], self._hidden_weight) 
        return RNNState.new_state([output_state, hidden_state])

    def _state_forward(self, inputs, state):
        if isinstance(inputs, (list, tuple)):
            raise Exception("Expecting rnn or gru state but got lstm state!")

        batch_size = inputs.size(1)
        output_state = state.repeat(1, batch_size, 1)
        output_state = F.dropout(output_state, p=self.dropout, 
                                 training=self.training)
        return output_state

    def initialize_parameters(self):
        nn.init.normal_(self._output_weight)
        if self.lstm_cell:
            nn.init.normal_(self._hidden_weight)

    def set_dropout(self, dropout):
        self._dropout = dropout
