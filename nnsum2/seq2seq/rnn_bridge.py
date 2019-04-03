from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.rnn_state import RNNState
import torch
import torch.nn as nn


@register_module("seq2seq.rnn_bridge")
class RNNBridge(Module):

    hparams = hparam_registry()

    @hparams()
    def num_layers(self):
        pass

    @hparams()
    def bidirectional_encoder(self):
        pass

    @hparams()
    def lstm_cell(self):
        pass
       
    @hparams(default=0.0)
    def dropout(self):
        pass

    @hparams()
    def encoder_dims(self):
        pass

    @hparams()
    def decoder_dims(self):
        pass

    @hparams(default=False)
    def pass_through(self):
        pass

    def init_network(self):
        if self.pass_through:
            return

        in_dims = self.encoder_dims
        if self.bidirectional_encoder:
            in_dims *= 2
        out_dims = self.decoder_dims
        
        self._output_state_networks = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dims, out_dims),
                          nn.Tanh(),
                          nn.Dropout(p=self.dropout))
            for _ in range(self.num_layers)])
        if self.lstm_cell:
            self._hidden_state_networks = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dims, out_dims),
                              nn.Tanh(),
                              nn.Dropout(p=self.dropout))
                for _ in range(self.num_layers)])

    def forward(self, inputs):
        if self.pass_through:
            return RNNState.new_state(inputs)
        elif self.lstm_cell:
            return self._lstm_forward(inputs)
        else: 
            return self._state_forward(inputs, self._output_state_networks)

    def _lstm_forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) and len(inputs) == 2:   
            raise Exception("Expecting lstm state!")
        output_state = self._state_forward(
            inputs[0], self._output_state_networks)
        hidden_state = self._state_forward(
            inputs[1], self._hidden_state_networks) 
        return RNNState.new_state([output_state, hidden_state])

    def _state_forward(self, inputs, networks):
        if isinstance(inputs, (list, tuple)):
            raise Exception("Expecting rnn or gru state but got lstm state!")

        batch_size = inputs.size(1)
            
        new_states = []
        if self.bidirectional_encoder:
            inputs = inputs.permute(1, 0, 2).contiguous().split(2, dim=1)
        else:
            inputs = inputs.permute(1, 0, 2).contiguous().split(1, dim=1)
        
        for net, inp in zip(networks, inputs):
            new_states.append(net(inp.view(batch_size, -1)))

        new_state = torch.stack(new_states)
        return new_state

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
        self._dropout = dropout
