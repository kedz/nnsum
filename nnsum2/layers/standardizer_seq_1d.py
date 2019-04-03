import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("layers.standardizer_seq_1d")
class StandardizerSeq1D(Module):
    """
    Normalize the embeddings of a sequence to be zero mean/unit variance
    across the time dimension. This module correctly handles masking so
    that the mean and standard deviation calculation ignore padding values.
    """
 
    hparams = hparam_registry()

    @hparams(default=True)
    def batch_first(self):
        pass

    @hparams(default=True)
    def unbiased(self):
        pass

    @hparams(default=1e-12)
    def eps(self):
        pass

    def initialize_parameters(self):
        pass

    def _compute_input_lengths(self, inputs, inputs_mask, time_dim):
        if inputs_mask is None:
            return inputs.size(time_dim)
        else:
            active_inputs = (~inputs_mask).float()
            return active_inputs.sum(time_dim, keepdim=True).unsqueeze(-1)

    def _center_inputs(self, inputs, inputs_mask, input_lengths, time_dim):
        masked_inputs = inputs.masked_fill(inputs_mask.unsqueeze(-1), 0)
        mean = masked_inputs.sum(time_dim, keepdim=True) / input_lengths
        centered_inputs = (inputs - mean).masked_fill(
            inputs_mask.unsqueeze(-1), 0)
        return centered_inputs
    
    def _compute_std(self, centered_inputs, input_lengths, time_dim):
        sum_of_squares = (centered_inputs ** 2).sum(time_dim, keepdim=True)
        if self.unbiased:
            return (sum_of_squares / (input_lengths - 1 + self.eps)).sqrt()
        else:
            return (sum_of_squares / (input_lengths + self.eps)).sqrt()

    def forward(self, inputs, inputs_mask=None):
        if self.batch_first:
            batch_dim = 0
            time_dim = 1
        else:
            batch_dim = 1
            time_dim = 0

        input_lengths = self._compute_input_lengths(inputs, inputs_mask,    
                                                    time_dim)
        centered_inputs = self._center_inputs(inputs, inputs_mask, 
                                              input_lengths, time_dim)
        std = self._compute_std(centered_inputs, input_lengths, time_dim)

        return centered_inputs / std, inputs_mask
