import torch.nn as nn
import torch.nn.functional as F


def target_getter(batch):
    return batch["target_output_features"]["tokens"], batch["target_lengths"]

class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_index=-1, target_getter=target_getter):
        super(CrossEntropyLoss, self).__init__()
        self.pad_index = pad_index
        self._target_getter = target_getter

    def forward(self, forward_state, targets):
        tgts, tgt_lens = self._target_getter(targets)

        target_logits = forward_state["target_logits"]

        total_xent = F.cross_entropy(
            target_logits.permute(1, 2, 0),
            tgts,
            ignore_index=self.pad_index,
            reduction="none")
        avg_xent = total_xent.sum() / tgt_lens.sum().float()
        return avg_xent
