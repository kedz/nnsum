import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers


@register_module("attention.feed_forward_kernel")
class FeedForwardKernel(Module):

    hparams = hparam_registry()

    @hparams(default=1.0)
    def temperature(self):
        pass

    @hparams()
    def hidden_dims(self):
        pass

    @property
    def is_stateless(self):
        return True

    def init_network(self):
        if isinstance(self.temperature, str): 
            if self.temperature != "auto":
                raise Exception(
                    "Temperature must be a non-negative float or 'auto'.")
        elif isinstance(self.temperature, (int, float)):
            if self.temperature <= 0:
                raise Exception(
                    "Temperature must be a non-negative float or 'auto'.")
        else:
            raise Exception(
                "Temperature must be a non-negative float or 'auto'.")
        self.weights = nn.Parameter(
            torch.FloatTensor(1, self.hidden_dims).normal_())

    def _apply_temperature(self, scores):
        if self.temperature == 1.0:
            return scores
        elif self.temperature == "auto":
            return scores / np.sqrt(self.hidden_dims)
        else:
            return scores / self.temperature

    def _broadcast_add(self, key, query):
        return key.unsqueeze(0) + query.unsqueeze(2)

    def forward(self, key, query, key_mask=None, state=None):
        # key = batch size x key length x embedding size
        # key_mask = batch size x key length
        # query = query length x batch size x embedding size

        # ff_inputs (query length x batch size x key length x embedding size)
        ff_inputs = torch.tanh(self._broadcast_add(key, query))

        # scores (query length x batch size x key length)
        scores = torch.nn.functional.linear(ff_inputs, self.weights).squeeze(3)

        # scores_temp (query length x batch size x key length)
        scores_temp = self._apply_temperature(scores)

        # If a mask is provided, set maked values to -inf such that 
        # 0 attention is given to them. 
        if key_mask is not None:
            scores_temp = scores_temp.masked_fill(
                key_mask.unsqueeze(0), float("-inf"))

        # attention (batch size x query length x key length)
        # attention is normalized across dimension 2 such that:
        # torch.allclose(attention.sum(2).eq(1.0)) == True
        attention = torch.softmax(scores_temp.permute(1, 0, 2), dim=2)

        return attention, state

    def initialize_parameters(self):
        nn.init.xavier_normal_(self.weights)

    def set_dropout(self, dropout):
        pass
