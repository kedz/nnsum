import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry
from .conv_seq_1d import ConvSeq1D
from .max_pooling_seq_1d import MaxPoolingSeq1D


@register_module("layers.conv_max_pooling_seq_1d")
class ConvMaxPoolingSeq1D(Module):
    """
    A 1 dimensional conv net for sequences.
    Architecture is:
       +-----------+    +--------------+    +----------+    +-------+
       |conv net 1d| -> |max pooling 1d| -> |activation| -> |dropout|
       +-----------+    +--------------+    +----------+    +-------+
    When using masking, this should be completely invariate to batching
    i.e. padding is correctly masked according to individal item sequence
    lengths and not the total batch length.
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

    @hparams(default="safe")
    def mask_mode(self):
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

    @hparams(default=None, required=False)
    def activation(self):
        pass

    @hparams(default=0.0, required=False)
    def dropout(self):
        pass

    def init_network(self):
        self._conv_layer = ConvSeq1D(
            input_features=self.input_features,
            output_features=self.output_features,
            kernel_width=self.kernel_width,
            padding=self.padding,
            batch_first=self.batch_first,
            bias=self.bias,
            output_mask_value=float("-inf"),
            mask_mode=self.mask_mode,
        )
        self._pooling_layer = MaxPoolingSeq1D(mask_mode="unsafe")

        if self.activation:
            self._act_func = nn.__dict__[self.activation]()
        else:
            self._act_func = None

    def initialize_parameters(self):
        self._conv_layer.initialize_parameters()

    def forward(self, inputs, inputs_mask=None):
        conv_out, conv_mask = self._conv_layer(inputs, inputs_mask=inputs_mask)
        pooling_out, pooling_mask = self._pooling_layer(
            conv_out, inputs_mask=conv_mask)
        
        if self._act_func:
            outputs = self._act_func(pooling_out)
        else:
            outputs = pooling_out

        if pooling_mask is not None:
            outputs.data.masked_fill_(pooling_mask.unsqueeze(-1), 0)

        outputs = F.dropout(outputs, p=self.dropout, training=self.training)

        return outputs, pooling_mask

    @property
    def kernel_weights(self):
        return self._conv_layer.kernel_weights

    @property
    def kernel_bias(self):
        return self._conv_layer.kernel_bias
