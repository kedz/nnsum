import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, logits_field="target_logits", 
                 target_field="target_labels",
                 log_probs_field="target_log_probability",
                 logits=True,
                 weights=None):
        super(CrossEntropyLoss, self).__init__()
        
        self._logits_field = logits_field
        self._log_probs_field = log_probs_field
        self._target_field = target_field
        self._total_loss = 0
        self._total_inputs = 0
        self._logits = logits
        self._weights = weights

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
    def log_probs_field(self):
        return self._log_probs_field

    @property
    def target_field(self):
        return self._target_field

    @property
    def logits(self):
        return self._logits

    def forward(self, forward_state, batch):
        if self.logits:
            return self._logits_forward(forward_state, batch)
        else:
            return self._log_probs_forward(forward_state, batch)

    def _logits_forward(self, forward_state, batch):
        target_logits = forward_state[self.logits_field]
        targets = batch[self.target_field]

        num_el = targets.size(0)
        avg_xent = F.cross_entropy(target_logits, targets, 
                                   weight=self._weights)
        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent

    def _log_probs_forward(self, forward_state, batch):

        target_log_probs = forward_state[self.log_probs_field]
        targets = batch[self.target_field]

        num_el = targets.size(0)
        avg_xent = F.nll_loss(target_log_probs, targets, weight=self._weights)

        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent
