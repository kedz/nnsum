import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry


@register_module("attention.no_mechanism")
class NoMechanism(Module):

    hparams = hparam_registry()

    @property
    def output_dims(self):
        return 0

    def forward(self, key, query, value=None, key_mask=None, state=None):
        return None, None, None

    def initialize_parameters(self):
        pass
