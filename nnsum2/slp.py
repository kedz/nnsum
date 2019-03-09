import torch.nn as nn
from .module import Module, register_module


@register_module("mlp")
class SingleLayerPerceptron(Module):

    hparams = Module.create_registry()

    @hparams(type="submodule")
    def input_layer(self):
        return self._input_layer
 
    @hparams(type="submodule")
    def output_layer(self):
        return self._output_layer
    
    def init_network(self):
        self._network = nn.Sequential(self._input_layer, self._output_layer)

    def forward(self, inputs):
        return self._network(inputs)
