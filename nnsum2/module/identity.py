from ..module import Module, register_module, hparam_registry


@register_module("modules.identity")
class Identity(Module):
    
    hparams = hparam_registry()

    def forward(self, inputs):
        return inputs

    def initialize_parameters(self):
        pass
