import torch
import torch.nn as nn
from collections import OrderedDict


class SequenceClassifier(nn.Module):
    def __init__(self, source_embedding_context, encoder,
                 target_embedding_context):
        super(SequenceClassifier, self).__init__()

        self._src_ec = source_embedding_context
        self._encoder = encoder
        self._tgt_ec = target_embedding_context

    @property
    def source_embedding_context(self):
        return self._src_ec

    @property
    def encoder(self):
        return self._encoder

    @property
    def target_embedding_context(self):
        return self._tgt_ec

    def initialize_parameters(self):
        print(" Initializing source embedding context parameters.")
        self.source_embedding_context.initialize_parameters()
        print(" Initializing encoder parameters.")
        self.encoder.initialize_parameters()
        print(" Initializing target embedding context parameters.")
        self.target_embedding_context.initialize_parameters()

    def forward(self, inputs):
        emb = self._src_ec(inputs["source_features"])
        rep = self._encoder(emb)
        logits = self._tgt_ec(rep)
        return logits  #, self._encoder.attention()
    
    def log_probs(self, inputs):
        all_logits = self.forward(inputs)
        all_log_probs = OrderedDict()
        for label, logits in all_logits.items():
            all_log_probs[label] = torch.log_softmax(logits, dim=1)
        return all_log_probs

    def predict_labels(self, inputs, return_attention=False):
        logits = self.forward(inputs)
        for logit in logits.values():
            bs = logit.size(0)
            break
        #bs = inputs["source_lengths"].size(0)
        pred_labels = [OrderedDict() for x in range(bs)]
        for cls, cls_logits in logits.items():
            vocab = self._tgt_ec.named_vocabs[cls]
            labels = cls_logits.max(1)[1]
            for i, lbl in enumerate(labels.tolist()):
                pred_labels[i][cls] = vocab[lbl]

        if return_attention:
            return pred_labels, self._encoder.attention()
        else:
            return pred_labels

    def score(self, inputs, labels):
        all_log_probs = self.log_probs(inputs)

        result = []
        for cls, log_probs in all_log_probs.items():
            result.append(log_probs.gather(1, labels[cls].view(-1,1)))
        return sum(result)
