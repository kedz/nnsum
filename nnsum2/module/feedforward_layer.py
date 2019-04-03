import torch.nn as nn
from ..module import Module, register_module, hparam_registry


@register_module("module.feedforward_layer")
class FeedForwardLayer(Module):
    
    hparams = hparam_registry()

    @hparams()
    def input_dims(self):
        pass

    @hparams()
    def output_dims(self):
        pass

    @hparams(default=0.0)
    def dropout(self):
        pass

    @hparams(default="Tanh")
    def activation(self):
        pass

    def init_network(self):
        self._network = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims),
            nn.__dict__[self.activation](),
            nn.Dropout(p=self.dropout),)

    def forward(self, inputs):
        return self._network(inputs)

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
