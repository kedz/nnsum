import torch
import torch.nn.functional as F
import torch.nn as nn
from ..module import Module, register_module, hparam_registry


@register_module("layers.max_pooling_seq_1d")
class MaxPoolingSeq1D(Module):
    """
    A 1 dimensional max pooling layer for sequences where the dimensions of 
    an input are (sequence length x batch size x embedding size) when 
    batch_first=False or (batch size x sequence length x embedding size) when
    batch_first=True.
    """

    hparams = hparam_registry()

    @hparams(default=None, required=False)
    def kernel_width(self):
        pass

    @hparams(default=True)
    def batch_first(self):
        pass

#    @hparams(default=1)
#    def stride(self):
#        pass

#    @hparams(default=1)
#    def dilation(self):
#        pass

    @hparams(default="safe")
    def mask_mode(self):
        pass

    @hparams(default=True)
    def squeeze_singleton(self):
        pass

    def initialize_parameters(self):
        pass

    def _apply_inputs_mask(self, inputs, inputs_mask):
        return inputs.masked_fill(inputs_mask.unsqueeze(2), float("-inf"))

    def _compute_outputs_mask(self, input_mask):
        if self.batch_first:
            batch_dim = 0 
            seq_dim = 1 
        else:
            batch_dim = 1 
            seq_dim = 0 

        batch_size = input_mask.size(batch_dim)
        seq_sizes = (~input_mask).float().sum(seq_dim)
        kernel_width = self.kernel_width if self.kernel_width else seq_sizes

        # Dilation and stride are currently hard coded.
        dilation = 1
        stride = kernel_width
        padding = 0
        output_seq_sizes = torch.floor(
          ( 
              seq_sizes 
            + padding * 2 
            - dilation * (kernel_width - 1) 
            - 1
          ) / stride 
          + 1
        )
        max_out_size = output_seq_sizes.max() + 1
        
        steps = torch.arange(1, max_out_size, device=input_mask.device)
        steps = steps.unsqueeze(0).repeat(batch_size, 1)
        outputs_mask = output_seq_sizes.view(-1, 1).lt(steps)
        if self.squeeze_singleton and outputs_mask.size(1) == 1:
            outputs_mask = outputs_mask.view(-1)
        return outputs_mask

    def forward(self, inputs, inputs_mask=None):

        if inputs.dim() != 3:
            raise ValueError(
                "inputs is expected to be a 3 dimensional tensor.")

        if inputs_mask is not None and self.mask_mode == "safe":
            inputs = self._apply_inputs_mask(inputs, inputs_mask)

        if not self.batch_first:
            inputs = inputs.permute(1, 0, 2)

        seq_size = inputs.size(1)
        emb_size = inputs.size(2)
        kernel_width = self.kernel_width if self.kernel_width else seq_size
        inputs = inputs.unsqueeze(1)
        outputs = F.max_pool2d(inputs, (kernel_width, 1))
        outputs = outputs.squeeze(1)

        if (outputs.size(1) == 1) and self.squeeze_singleton:
            outputs = outputs.squeeze(1)

        if inputs_mask is None:
            outputs_mask = None
        else:
            outputs_mask = self._compute_outputs_mask(inputs_mask)
            outputs.data.masked_fill_(outputs_mask.unsqueeze(-1), 0)

        return outputs, outputs_mask
