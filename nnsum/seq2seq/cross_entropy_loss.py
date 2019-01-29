import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.pad_index = pad_index

    def forward(self, forward_state, targets, target_lengths):
        target_logits = forward_state["target_logits"]

        total_xent = F.cross_entropy(
            target_logits.permute(1, 2, 0),
            targets,
            ignore_index=self.pad_index,
            reduction="none")
        avg_xent = total_xent.sum() / target_lengths.sum().float()
        return avg_xent









        steps, bsize, vsize = target_logits.size()
        target_logits_flat = target_logits.contiguous().view(
            steps * bsize, vsize)
        targets_flat = targets.t().contiguous().view(-1)
        

        # TODO test this. I think its wrong.
        total_xent = F.cross_entropy(
            target_logits_flat,
            targets_flat,
            ignore_index=self.pad_index,
            reduction="sum")

        avg_xent = total_xent / target_lengths.sum().float()

        return avg_xent
