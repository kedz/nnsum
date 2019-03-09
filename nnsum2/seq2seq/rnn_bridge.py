from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.rnn_state import RNNState
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

    def init_network(self):
        
        total_nets = self.num_layers
        if self.bidirectional_encoder:
            total_nets *= 2
        if self.lstm_cell:
            total_nets *= 2
        
        nets = []
        for _ in range(total_nets):
            net = nn.Sequential(
                nn.Linear(self.encoder_dims, self.decoder_dims),
                nn.Tanh(),
                nn.Dropout(p=self.dropout))
            nets.append(net)
        self._networks = nn.ModuleList(nets)


    def forward(self, inputs):
        print()
        print(inputs)

        if self.lstm_cell:
            return self._lstm_forward(inputs)
        else: 
            return self._rnn_forward(inputs)

    def _lstm_forward(self, inputs):
        pass

    def _rnn_forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            raise Exception("Expecting rnn or gru state but got lstm state!")
            
        split_inputs = inputs.split(1, dim=0)
        print(len(split_inputs))
        input()


    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)    
            else:
                nn.init.normal_(param)           
