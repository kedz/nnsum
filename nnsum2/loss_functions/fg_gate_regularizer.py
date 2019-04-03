import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.fg_gate_regularizer")
class FGGateRegularizer(Module):

    hparams = hparam_registry()

    @hparams(default="gates")
    def gates_field(self):
        pass

    @hparams(default="target_labels")
    def target_field(self):
        pass

    @hparams(default=5.)
    def active_target(self):
        pass

    @hparams(default=0.)
    def inactive_target(self):
        pass

    @hparams()
    def label_vocab(self):
        pass

    @hparams(default=None, required=False)
    def label_name(self):
        pass

    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def init_network(self):
        self.reset()
        self._has_na = "N/A" in self.label_vocab
        if self._has_na:
            self._na_index = self.label_vocab["N/A"]

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def forward(self, forward_state, batch):

        total_activation = forward_state[self.gates_field].sum(1)

        target_activation = total_activation.new(total_activation.size())
        target_activation.fill_(self.active_target)

        if self._has_na:        
            targets = batch[self.target_field]
            if self.label_name is not None:
                targets = targets[self.label_name]

            target_activation = target_activation.masked_fill(
                targets.eq(self._na_index), 
                self.inactive_target)

        loss_el = (target_activation - total_activation) ** 2
        avg_loss = loss_el.mean()
        self._total_loss += (avg_loss.item() * loss_el.size(0))
        self._total_inputs += loss_el.size(0)

        return avg_loss
