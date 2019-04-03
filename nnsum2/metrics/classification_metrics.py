from ..parameterized import Parameterized
from ..hparam_registry import HParams

from sklearn.metrics import precision_recall_fscore_support
import numpy as np


@Parameterized.register_object("metrics.classification_metrics")
class ClassificationMetrics(Parameterized):
    
    hparams = HParams()

    @hparams(default=True)
    def use_logits(self):
        pass

    @hparams(default="target_logits")
    def logits_field(self):
        pass

    @hparams(default="target_log_probability")
    def log_probs_field(self):
        pass

    @hparams(default="target_labels")
    def target_field(self):
        pass

    @hparams(default=None, required=False)
    def target_name(self):
        pass

    @hparams()
    def label_vocab(self):
        pass

    def init_object(self):
        self._true_labels = []
        self._pred_labels = []
    
    def init_network(self):
        self.reset()
 
    def reset(self):
        self._true_labels = []
        self._pred_labels = []
        self._cache = None

    def compute(self):
        if len(self._true_labels) > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def __call__(self, forward_state, batch):
        if self.use_logits:
            return self._logits_forward(forward_state, batch)
        else:
            return self._log_probs_forward(forward_state, batch)

    def _logits_forward(self, forward_state, batch):
        self._pred_labels.extend(
            forward_state[self.logits_field].max(1)[1].tolist())
        targets = batch[self.target_field]
        if self.target_name is not None:
            targets = targets[self.target_name]
        self._true_labels.extend(targets.tolist())

    def _log_probs_forward(self, forward_state, batch):

        self._pred_labels.extend(
            forward_state[self.log_probs_field].max(1)[1].tolist())
        targets = batch[self.target_field]
        if self.target_name is not None:
            targets = targets[self.target_name]
        self._true_labels.extend(targets.tolist())

    def compute(self):
        labels = [label for label in self.label_vocab]
        true_labels = [self.label_vocab[y] for y in self._true_labels] 
        pred_labels = [self.label_vocab[y] for y in self._pred_labels] 
         
        prec, recall, fscore, supp = precision_recall_fscore_support(
            true_labels, pred_labels, labels=labels)

        results = {}
        for i, label in enumerate(labels):
            if supp[i] > 0:
                results[label] = {"precision": prec[i],
                                  "recall": recall[i],
                                  "fscore": fscore[i]}
        avg_prec = np.mean([x for i, x in enumerate(prec) if supp[i] > 0])   
        avg_recall = np.mean([x for i, x in enumerate(recall) if supp[i] > 0])
        avg_fscore = np.mean([x for i, x in enumerate(fscore) if supp[i] > 0])
        results["AVERAGE"] = {"precision": avg_prec,
                              "recall": avg_recall,
                              "fscore": avg_fscore}
        self._cache = results
        return results

    def pretty_print(self):
        if self._cache is None:
            results = self.compute()
        else:
            results = self._cache
        max_label_size = str(max([len(label) for label in self.label_vocab]))
        template = (
            "LABEL={:" + max_label_size + "s} PREC={precision:5.3f}  "
            + "RECALL={recall:5.3f}  FSCORE={fscore:5.3f}"
        )
        for label in self.label_vocab:
            if label not in results:
                continue
            print(template.format(label, **results[label]))
        print(template.format("AVERAGE", **results["AVERAGE"]))
