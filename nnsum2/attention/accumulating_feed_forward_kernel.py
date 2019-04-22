import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry
import numpy as np
import nnsum2.layers


@register_module("attention.accumulating_feed_forward_kernel")
class AccumulatingFeedForwardKernel(Module):

    hparams = hparam_registry()

    @hparams(default=1.0)
    def temperature(self):
        pass

    @hparams()
    def hidden_dims(self):
        pass

    @property
    def is_stateless(self):
        return False

    @hparams(default=True)
    def learn_accumulator_weights(self):
        pass

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

        if self.learn_accumulator_weights:
            self.accumulator_weight = nn.Parameter(
                torch.FloatTensor(1, 1, 1, self.hidden_dims).normal_())
        else:
            self.register_parameter('accumulator_weight', None)

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

        if state is None or state.get("accumulator", None) is None:
            accumulator = key.new(
                key.size(0), 1, key.size(1), 1).zero_()
        else:
            accumulator = state["accumulator"].unsqueeze(3)

        # hidden_states (query length x batch size x key length x embedding size)
        hidden_states = self._broadcast_add(key, query)
        
        attention = []
        for hidden_state in hidden_states.permute(1, 0, 2, 3).split(1, dim=1):
            if self.learn_accumulator_weights:
                ff_inputs = torch.tanh(
                    hidden_state - self.accumulator_weight * accumulator)
            else:
                ff_inputs = torch.tanh(
                    hidden_state - accumulator)
            # scores (batch x 1 x key length)
            scores = torch.nn.functional.linear(
                ff_inputs, self.weights).squeeze(3)
            scores_temp = self._apply_temperature(scores)

            # If a mask is provided, set maked values to -inf such that 
            # 0 attention is given to them. 
            if key_mask is not None:
                scores_temp = scores_temp.masked_fill(
                    key_mask.unsqueeze(1), float("-inf"))
 
            attention_step = torch.softmax(scores_temp, dim=2)
            accumulator = accumulator + attention_step.unsqueeze(3)
            attention.append(attention_step)

        # attention (batch size x query length x key length)
        # attention is normalized across dimension 2 such that:
        # torch.allclose(attention.sum(2).eq(1.0)) == True
        attention = torch.cat(attention, dim=1)
        accumulator = accumulator.squeeze(3)
        return attention, {"accumulator": accumulator}

    def initialize_parameters(self):
        nn.init.xavier_normal_(self.weights)
        nn.init.xavier_normal_(self.accumulator_weight)

    def set_dropout(self, dropout):
        pass
