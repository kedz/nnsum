import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.0, 
                 initial_weights=None, trainable=True):

        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=0)
        self.dropout_ = dropout
        self.trainable_ = trainable

        if initial_weights is not None:
            if (vocab_size != initial_weights.size(0)) \
                    or (embedding_size != initial_weights.size(1)) \
                    or initial_weights.dim() != 2:
                raise Exception(
                    ("EmbbedingLayer: initial_weights has dims: ({}) "
                     " but expected ({}, {}).").format(
                        ",".join([str(s) for s in initial_weights.size()]),
                        vocab_size, embedding_size))
            self.embeddings.weight.data.copy_(initial_weights)

    def parameters(self):
        if self.trainable_:
            for p in self.embeddings.parameters():
                yield p

    def named_parameters(self, memo, submod_prefix):
        if self.trainable_:
            for n, p in self.embeddings.named_parameters(memo, submod_prefix):
                yield n, p

    def forward(self, inputs):
        if inputs.dim() == 2:
            emb = self.embeddings(inputs)
        else:
            bs = inputs.size(0)
            ss = inputs.size(1)
            ts = inputs.size(2)
            inputs_flat = inputs.view(bs * ss, ts)
            emb = self.embeddings(inputs_flat).view(bs, ss, ts, -1)
        emb = F.dropout(emb, p=self.dropout_, training=self.training)
        return emb

    @property
    def size(self):
        return self.embeddings.weight.size(1)
