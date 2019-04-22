from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.precomputed")
class Precomputed(Module):

    hparams = hparam_registry()

    @hparams()
    def field(self):
        pass

    def init_network(self):
        self.reset()
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def forward(self, forward_state, batch):
        loss = forward_state[self.field]
        self._total_loss += loss.item()
        self._total_inputs += 1
        return loss

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")
