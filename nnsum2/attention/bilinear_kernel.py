import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers


@register_module("attention.bilinear_kernel")
class BiLinearKernel(Module):

    hparams = hparam_registry()

    @hparams(default=1.0)
    def temperature(self):
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

    def _apply_temperature(self, scores, dim_size):
        if self.temperature == 1.0:
            return scores
        elif self.temperature == "auto":
            return scores / np.sqrt(dim_size)
        else:
            return scores / self.temperature

    def forward(self, key, query, key_mask=None, state=None):
        # key = batch size x key length x embedding size
        # key_mask = batch size x key length
        # query = query length x batch size x embedding size

        # scores (batch size x query length x key length
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(0, 2, 1))

        # scores_temp (batch size x query length x key length)
        embedding_size = key.size(2)
        scores_temp = self._apply_temperature(scores, embedding_size)

        # If a mask is provided, set maked values to -inf such that 
        # 0 attention is given to them. 
        if key_mask is not None:
            scores_temp = scores_temp.masked_fill(
                key_mask.unsqueeze(1), float("-inf"))

        # attention (batch size x query length x key length)
        # attention is normalized across dimension 2 such that:
        # torch.allclose(attention.sum(2).eq(1.0)) == True
        attention = torch.softmax(scores_temp, dim=2)

        return attention, None

    def initialize_parameters(self):
        pass
