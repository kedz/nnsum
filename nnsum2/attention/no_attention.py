from ..module import Module, register_module, hparam_registry


@register_module("attention.no_attention")
class NoAttention(Module):

    hparams = hparam_registry()

    @hparams(default=True)
    def compute_composition(self):
        pass

    def forward(self, context, query, context_mask=None, attention_state=None):
        if self.compute_composition:
            return None, None, None
        else:
            return None, None
