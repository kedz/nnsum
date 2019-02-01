import torch
import torch.nn as nn


def target_getter(batch):
    return (batch["copy_targets"],
            batch["target_output_features"]["tokens"], 
            batch["target_lengths"])

class PointerGeneratorCrossEntropyLoss(nn.Module):
    def __init__(self, pad_index=-1, target_getter=target_getter):
        super(PointerGeneratorCrossEntropyLoss, self).__init__()
        self.pad_index = pad_index
        self._target_getter = target_getter

    def forward(self, forward_state, targets):
        ptr_tgts, gen_tgts, tgt_len = self._target_getter(targets)

        ptr_tgts = ptr_tgts.t().unsqueeze(-1)
        gen_tgts = gen_tgts.t().unsqueeze(-1)
        no_gen_mask = gen_tgts != ptr_tgts
        pad_mask = gen_tgts.eq(self.pad_index)

        ptr_probs = forward_state["pointer_probability"].gather(2, ptr_tgts)
        gen_probs = forward_state["generator_probability"].gather(2, gen_tgts)
        gen_probs = gen_probs.masked_fill(no_gen_mask, 0.)

        log_probs = torch.log(gen_probs + ptr_probs)
        log_probs.data.masked_fill_(pad_mask, 0.)
        total_log_probs = log_probs.sum()
        avg_xent = -total_log_probs / tgt_len.sum().float()
        return avg_xent
