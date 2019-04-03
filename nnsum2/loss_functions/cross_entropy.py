import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.cross_entropy")
class CrossEntropy(Module):

    hparams = hparam_registry()

    @hparams(default=True)
    def use_logits(self):
        pass

    @hparams(default="target_logits")
    def logits_field(self):
        pass

    @hparams(default="target_log_probability")
    def log_probs_field(self):
        pass

    @hparams(default="target_labels")
    def target_field(self):
        pass

    @hparams(default=None, required=False)
    def label_name(self):
        pass


    def init_network(self):
        self.reset()
        self._weights = None
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def forward(self, forward_state, batch):
        if self.use_logits:
            return self._logits_forward(forward_state, batch)
        else:
            return self._log_probs_forward(forward_state, batch)

    def _logits_forward(self, forward_state, batch):
        target_logits = forward_state[self.logits_field]
        targets = batch[self.target_field]
        if self.label_name is not None:
            targets = targets[self.label_name]

        num_el = targets.size(0)
        avg_xent = F.cross_entropy(target_logits, targets, 
                                   weight=self._weights)
        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent

    def _log_probs_forward(self, forward_state, batch):

        target_log_probs = forward_state[self.log_probs_field]
        targets = batch[self.target_field]
        if self.label_name is not None:
            targets = targets[self.label_name]

        num_el = targets.size(0)
        avg_xent = F.nll_loss(target_log_probs, targets, weight=self._weights)

        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent
