import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers

from .attention_interface_v1 import KeyValueQueryInterface
from .feed_forward_kernel import FeedForwardKernel
from .accumulating_feed_forward_kernel import AccumulatingFeedForwardKernel


@register_module("attention.feed_forward_mechanism")
class FeedForwardMechanism(Module):

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
    
    @hparams()
    def key_dims(self):
        pass

    @hparams()
    def query_dims(self):
        pass

    @hparams()
    def hidden_dims(self):
        pass

    @hparams(default=True)
    def learn_accumulator_weights(self):
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
            kernel = AccumulatingFeedForwardKernel(
                temperature=self.temperature, hidden_dims=self.hidden_dims,
                learn_accumulator_weights=self.learn_accumulator_weights)
        else:
            kernel = FeedForwardKernel(
                temperature=self.temperature, hidden_dims=self.hidden_dims)

        self._mechanism = KeyValueQueryInterface(
            kernel=kernel,
            key_adaptor=nnsum2.layers.FullyConnected(
                in_feats=self.key_dims, out_feats=self.hidden_dims,
                bias=True, activation="none"),
            query_adaptor=nnsum2.layers.FullyConnected(
                in_feats=self.query_dims, out_feats=self.hidden_dims,
                bias=False, activation="none"),
        )

    def forward(self, key, query, value=None, key_mask=None, state=None):
        return self.mechanism(
            key, query, value=value, key_mask=key_mask, state=state)

    def initialize_parameters(self):
        self.mechanism.initialize_parameters()

    def set_dropout(self, dropout):
        self.mechanism.set_dropout(dropout)
