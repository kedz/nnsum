from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import torch


class ClassificationMetrics(Metric):
    def __init__(self, vocab, output_transform=lambda x: x):
        label_size = len(vocab)
        self._confusion = torch.zeros([label_size] * 2, dtype=torch.float32)
        self._vocab = vocab
        super(ClassificationMetrics, self).__init__(
            output_transform=output_transform)

    def reset(self):
        self._confusion.fill_(0)
        self._num_examples = 0

    def update(self, output):
        for true_label, pred_label in zip(*output):
            if true_label == -1:
                continue
            self._confusion[true_label, pred_label] += 1
            self._num_examples += 1
    
    def compute(self):

        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can '
                'be computed')

        tp = torch.diag(self._confusion)
        label_counts = self._confusion.sum(1)
        nz_labels = label_counts.ne(0).float().sum()

        acc = (tp.sum() / self._num_examples).item()
        fp = self._confusion.sum(0) - tp
        fn = self._confusion.sum(1) - tp

        tp_and_fp = tp + fp
        tp_and_fp.masked_fill_(tp_and_fp == 0, 1)
        prec = (tp / tp_and_fp)

        tp_and_fn = tp + fn
        tp_and_fn.masked_fill_(tp_and_fn == 0, 1)
        recall = (tp / tp_and_fn)

        prec_and_recall = prec + recall
        prec_and_recall.masked_fill_(prec_and_recall == 0, 1.)
        f1 = 2 * prec * recall / prec_and_recall
        macro_avg_f1 = f1.sum().item() / nz_labels
        f1 = f1.tolist()
        f1 = {label: f1[i] for i, label in self._vocab.enumerate()}
        f1["macro avg."] = macro_avg_f1

        macro_avg_prec = prec.sum().item() / nz_labels
        prec = prec.tolist()
        prec = {label: prec[i] for i, label in self._vocab.enumerate()}
        prec["macro avg."] = macro_avg_prec

        macro_avg_recall = recall.sum().item() / nz_labels
        recall = recall.tolist()
        recall = {label: recall[i] for i, label in self._vocab.enumerate()}
        recall["macro avg."] = macro_avg_recall

        return {"accuracy": acc, "precision": prec, "recall": recall,
                "f-measure": f1}
