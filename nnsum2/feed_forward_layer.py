import torch.nn as nn
from .module import Module, register_module, hparam_registry


@register_module("fflayer")
class FeedForwardLayer(Module):
    
    hparams = hparam_registry()

    @hparams(default=100)
    def input_dims(self):
        pass

    @hparams(default=100)
    def output_dims(self):
        pass

    @hparams(default="ReLU")
    def activation(self):
        pass

    @hparams(default=.25)
    def dropout(self):
        pass

    @hparams(default=True)
    def dropout_final(self):
        pass

    def init_network(self):

        linear = nn.Linear(self.input_dims, self.output_dims)
        activation = nn.__dict__[self.activation]()
        dropout_inplace = not self.dropout_final and self.activation == "ReLU"
        dropout = nn.Dropout(p=self.dropout, inplace=dropout_inplace)

        if self.dropout_final:
            self._network = nn.Sequential(linear, activation, dropout)
        else:
            self._network = nn.Sequential(linear, dropout, activation)
        
        return self._network

    def forward(self, inputs):
        return self._network(inputs)
