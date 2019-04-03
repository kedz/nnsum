import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.attention_continuity")
class AttentionContinuity(Module):

    hparams = hparam_registry()

    @hparams(default="context_attention")
    def attention_field(self):
        pass

    def init_network(self):
        self.reset()
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def forward(self, forward_state, batch):
        
        attention = forward_state[self.attention_field]
        attn_unfold = attention.unfold(0,2,1)
        mean = attn_unfold.mean(3, keepdim=True).detach()
        dist = ((attn_unfold - mean) **2).mean(3)
        avg_loss = dist.mean()

        self._total_loss += (avg_loss.item() * dist.numel())
        self._total_inputs += dist.numel()

        return avg_loss
