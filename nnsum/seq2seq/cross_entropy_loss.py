import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, logits_field="target_logits", 
                 target_field="target_output_features",
                 target_mask_field="target_mask"):
        super(CrossEntropyLoss, self).__init__()
        
        self._logits_field = logits_field
        self._target_field = target_field
        self._target_mask_field = target_mask_field
        self._total_loss = 0
        self._total_inputs = 0

    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    @property
    def logits_field(self):
        return self._logits_field

    @property
    def target_field(self):
        return self._target_field

    @property
    def target_mask_field(self):
        return self._target_mask_field

    def forward(self, forward_state, batch):
        
        targets = batch[self.target_field]
        assert len(targets) == 1
        targets = list(targets.values())[0]

        target_logits = forward_state[self.logits_field]
        steps, batches, vsize = target_logits.size()
        logits_flat = target_logits.view(steps * batches, vsize)

        targets_flat = targets.t().contiguous().view(-1)

        el_xent_flat = F.cross_entropy(logits_flat, targets_flat,
                                       reduction="none")
        el_xent = el_xent_flat.view(steps, batches)

        tgt_mask = batch.get(self.target_mask_field, None)
        if tgt_mask is not None:
            el_xent = el_xent.masked_fill(tgt_mask.t(), 0.)
            num_el = batches * steps - tgt_mask.long().sum().item()
        else:
            num_el = batches * steps

        total_xent = el_xent.sum()

        self._total_loss += total_xent.item()
        self._total_inputs += num_el

        return total_xent / num_el
