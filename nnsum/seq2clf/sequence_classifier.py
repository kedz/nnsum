import torch
import torch.nn as nn
from collections import OrderedDict


class SequenceClassifier(nn.Module):
    def __init__(self, encoder, predictor):
        super(SequenceClassifier, self).__init__()

        self._encoder = encoder
        self._predictor = predictor

    @property
    def encoder(self):
        return self._encoder
    
    @property
    def predictor(self):
        return self._predictor

    def initialize_parameters(self):
        print(" Initializing encoder parameters.")
        self.encoder.initialize_parameters()
        print(" Initializing predictor parameters.")
        self.predictor.initialize_parameters()

    def forward(self, inputs):
        encoder_output, encoder_state = self.encoder(
            inputs["source_input_features"],
            inputs["source_lengths"])
        logits = self.predictor(encoder_output)
        return {"target_logits": logits, "encoder_state": encoder_state}
    
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
