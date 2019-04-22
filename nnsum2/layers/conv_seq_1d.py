import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("layers.conv_seq_1d")
class ConvSeq1D(Module):
    """
    A 1 dimensional convolutional layer for sequences where the dimensions of 
    an input are (sequence length x batch size x embedding size) when 
    batch_first=False or (batch size x sequence length x embedding size) when
    batch_first=True.
    """

    hparams = hparam_registry()

    @hparams()
    def kernel_width(self):
        pass

    @hparams()
    def input_features(self):
        pass

    @hparams()
    def output_features(self):
        pass

    @hparams(default=True)
    def batch_first(self):
        pass

    @hparams(default=0, required=False)
    def padding(self):
        pass

#    @hparams(default=1)
#    def stride(self):
#        pass

    @hparams(default=True)
    def bias(self):
        pass

#    @hparams(default=1)
#    def dilation(self):
#        pass

    @hparams(default=0)
    def output_mask_value(self):
        pass

    @hparams(default="safe")
    def mask_mode(self):
        pass

    @property
    def kernel_weights(self):
        return self._network.weight

    @property
    def kernel_bias(self):
        return self._network.bias

    @hparams(default=None, required=False)
    def activation(self):
        pass

    @hparams(default=0.0, required=False)
    def dropout(self):
        pass

    def init_network(self):
        kernel_size = (self.kernel_width, self.input_features) 
        padding = (self.padding, 0)
        self._network = nn.Conv2d(1, self.output_features, kernel_size,
                                  padding=padding, bias=self.bias)

        if self.activation:
            self._act_func = nn.__dict__[self.activation]()
        else:
            self._act_func = None


    def initialize_parameters(self):
        k = self.kernel_width
        nn.init.xavier_normal_(self.kernel_weights)
        nn.init.uniform_(self.kernel_bias, a=-k, b=k)

    def _apply_inputs_mask(self, inputs, inputs_mask):
        return inputs.masked_fill(inputs_mask.unsqueeze(2), 0)

    def _compute_outputs_mask(self, input_mask):
        if self.batch_first:
            batch_dim = 0 
            seq_dim = 1 
        else:
            batch_dim = 1 
            seq_dim = 0 

        batch_size = input_mask.size(batch_dim)
        seq_sizes = (~input_mask).float().sum(seq_dim)

        # Dilation and stride are currently hard coded.
        dilation = 1
        stride = 1
        output_seq_sizes = torch.floor(
          ( 
              seq_sizes 
            + self.padding * 2 
            - dilation * (self.kernel_width - 1) 
            - 1
          ) / stride 
          + 1
        )
        max_out_size = output_seq_sizes.max() + 1
        
        steps = torch.arange(1, max_out_size, device=input_mask.device)
        steps = steps.unsqueeze(0).repeat(batch_size, 1)
        outputs_mask = output_seq_sizes.view(-1, 1).lt(steps)
        return outputs_mask

    def forward(self, inputs, inputs_mask=None):

        if inputs.dim() != 3:
            raise ValueError(
                "inputs is expected to be a 3 dimensional tensor.")

        if inputs_mask is not None and self.mask_mode == "safe":
            inputs = self._apply_inputs_mask(inputs, inputs_mask)

        if not self.batch_first:
            inputs = inputs.permute(1, 0, 2)

        inputs = inputs.unsqueeze(1)
        outputs = self._network(inputs).squeeze(3).permute(0, 2, 1)

        if self._act_func:
            outputs = self._act_func(outputs)

        outputs = F.dropout(outputs, p=self.dropout, training=self.training)

        if inputs_mask is not None:
            outputs_mask = self._compute_outputs_mask(inputs_mask)
            outputs = outputs.masked_fill(
                outputs_mask.unsqueeze(2), self.output_mask_value)
        else: 
            outputs_mask = None

        return outputs, outputs_mask

    def set_dropout(self, dropout):
        self._dropout = dropout
