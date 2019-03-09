import torch
import torch.nn as nn
from ..module import Module, register_module, hparam_registry


@register_module("seq2seq.passthrough_encoder")
class PassThroughEncoder(Module):
    hparams = hparam_registry()

    @hparams(type="embedding_context")
    def embedding_context(self):
        pass

  
#    @hparams()
#    def init_state_dims(self):
#        pass
#
#    @hparams(default=0.)
#    def init_state_dropout(self):
#        pass
#
    def init_network(self):
        pass

#        self.init_state = torch.nn.Parameter(
#            torch.FloatTensor(1, self.init_state_dims).normal_())
#        self._init_state_dropout_module = nn.Dropout(p=self.init_state_dropout)

    def forward(self, features, lengths):
        emb = self.embedding_context(features)
        batch_size = emb.size(1)
        state = self.init_state.repeat(batch_size, 

