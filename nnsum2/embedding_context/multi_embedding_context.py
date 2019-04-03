import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("multi_embedding_context")
class MultiEmbeddingContext(Module):

    hparams = hparam_registry()

    @hparams()
    def embedding_contexts(self):
        pass

    @property
    def named_vocabs(self):
        return {name: ec.vocab for name, ec in self.embedding_contexts.items()}

#i#    def parameters(self):
 #       for ec in self.embedding_contexts.values():
#            for param in ec.parameters():
#                yield param

    def init_network(self):
        self._ecs = nn.ModuleList([ec for ec in self.embedding_contexts.values()])

#    def cuda(self, device):
#        for name in self.embedding_contexts.keys():
#            self.embedding_contexts[name] = \
#                self.embedding_contexts[name].cuda(device)
#        return self

    def initialize_parameters(self):
        for ec in self._ecs:
            ec.initialize_parameters()
            
    def forward(self, inputs):
        
        if not isinstance(inputs, dict):
            raise Exception("Input must be a dict")

        outputs = []
        for ec in self._ecs:
            outputs.append(ec(inputs[ec.name]))
        return torch.cat(outputs, 2)
            
