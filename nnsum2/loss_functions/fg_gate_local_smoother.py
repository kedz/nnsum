import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.fg_gate_local_smoother")
class FGGateLocalSmoother(Module):

    hparams = hparam_registry()

    @hparams(default="gates")
    def gates_field(self):
        pass

    @hparams(default=0)
    def padding(self):
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

        gates = forward_state[self.gates_field]
        
        if self.padding > 0:
            gates = F.pad(gates, (self.padding, self.padding), "constant", 0)
        
        gates_unfolded = gates.unfold(1, 2, 1)
        mean = gates_unfolded.mean(2, keepdim=True).detach()
        loss_el = ((gates_unfolded - mean) ** 2).sum(2).sum(1)
        avg_loss = loss_el.mean()
        self._total_loss += (avg_loss.item() * loss_el.size(0))
        self._total_inputs += loss_el.size(0)

        return avg_loss
