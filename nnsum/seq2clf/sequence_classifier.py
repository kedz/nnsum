import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(self, source_embedding_context, encoder,
                 target_embedding_context):
        super(SequenceClassifier, self).__init__()

        self._src_ec = source_embedding_context
        self._encoder = encoder
        self._tgt_ec = target_embedding_context

    def forward(self, inputs):
        emb = self._src_ec(inputs["source_features"])
        rep = self._encoder(emb)
        logits = self._tgt_ec(rep)
        return logits
