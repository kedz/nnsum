from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.rnn_state import RNNState


@register_module("seq2seq.rnn_encoder")
class RNNEncoder(Module):

    hparams = hparam_registry()

    @hparams()
    def embedding_context(self):
        pass

    @hparams()
    def rnn(self):
        pass

    @hparams()
    def bridge(self):
        pass

    def forward(self, features, lengths):
        emb = self.embedding_context(features)
        context, state = self.rnn(emb, lengths=lengths)
        return context.permute(1, 0, 2), self.bridge(state)

    def initialize_parameters(self):
        self.rnn.initialize_parameters()
        self.embedding_context.initialize_parameters()
        self.bridge.initialize_parameters()

    def set_dropout(self, dropout):
        self.embedding_context.set_dropout(dropout)
        self.rnn.set_dropout(dropout)
        self.bridge.set_dropout(dropout)
