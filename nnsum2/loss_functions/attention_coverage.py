import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.attention_coverage")
class AttentionCoverage(Module):

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
        tgt = 1
        mask = batch["source_mask"]
        if "max_references" in batch:
            batch_size = mask.size(0)
            max_refs = batch["max_references"]
            mask = mask.unsqueeze(1).repeat(1, max_refs, 1)
            mask = mask.view(batch_size * max_refs, -1)
            source_lengths = batch["source_lengths"].view(-1, 1)\
                .repeat(1, max_refs).view(batch_size * max_refs).float()
            target_mask = batch["target_lengths"].eq(0).view(-1, 1).repeat(
                1, mask.size(1))
            mask = mask | target_mask
        else:
            source_lengths = batch["source_lengths"].float()
            target_mask = batch["target_lengths"].eq(0).view(-1, 1).repeat(
                1, mask.size(1))
            mask = mask | target_mask

        attention = forward_state[self.attention_field].sum(0)
        attention_clamped = attention.clamp(0, tgt)
        el_loss = ((attention_clamped - tgt) ** 2).masked_fill(mask, 0)
        
        # Ignore attending to the start symbol.
        el_loss = el_loss[:,1:]
        
        loss = el_loss.sum(1) / source_lengths

        avg_loss = loss.mean()
        self._total_loss += (avg_loss.item() * attention.size(0))
        self._total_inputs += attention.size(0)

        return avg_loss
