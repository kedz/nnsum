import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers

from .attention_interface_v1 import KeyValueQueryInterface
from .bilinear_kernel import BiLinearKernel
from .accumulating_bilinear_kernel import AccumulatingBiLinearKernel


@register_module("attention.bilinear_mechanism")
class BiLinearMechanism(Module):

    hparams = hparam_registry()

    @hparams(default=False)
    def accumulate(self):
        pass

    @hparams(default=1.0)
    def temperature(self):
        pass

    @hparams(default=None, required=False)
    def value_dims(self):
        pass

    @property
    def mechanism(self):
        return self._mechanism

    @property
    def output_dims(self):
        if self.value_dims is None:
            raise Exception(
                "value_dims is not set so output dims can't be inferred.")
        else:
            return self.value_dims

    def init_network(self):
        if self.accumulate:
            kernel = AccumulatingBiLinearKernel(temperature=self.temperature)
        else:
            kernel = BiLinearKernel(temperature=self.temperature)
        self._mechanism = KeyValueQueryInterface(kernel=kernel)

    def forward(self, key, query, value=None, key_mask=None, state=None):
        return self.mechanism(
            key, query, value=value, key_mask=key_mask, state=state)

    def initialize_parameters(self):
        self.mechanism.initialize_parameters()
