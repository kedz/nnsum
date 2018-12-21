import torch
import nnsum


def test_accuracy():
    
    vocab = nnsum.embedding_context.Vocab.from_word_list(
        ["negative", "positive", "neutral"], pad=None, unk=None)
    
    metric = nnsum.metrics.ClassificationMetrics(vocab)

    true_labels = torch.LongTensor(100).random_(0, len(vocab))
    pred_labels = torch.LongTensor(100).random_(0, len(vocab))
    
    ref_acc = (true_labels == pred_labels).float().sum() / true_labels.size(0)
    ref_acc = ref_acc.item()
    
    metric.update((true_labels, pred_labels))

    assert metric.compute()["accuracy"] == ref_acc

def test_precision_recall_fmeasure():
    
    vocab = nnsum.embedding_context.Vocab.from_word_list(
        ["negative", "positive", "neutral"], pad=None, unk=None)
    

    true_labels = torch.LongTensor(100).random_(0, len(vocab))
    pred_labels = torch.LongTensor(100).random_(0, len(vocab))
    metric = nnsum.metrics.ClassificationMetrics(vocab)
    metric.update((true_labels, pred_labels))

    precision = []
    recall = []
    for i, label in vocab.enumerate():
        tp = ((true_labels == i) & (pred_labels == i)).float().sum()
        fp = ((true_labels != i) & (pred_labels == i)).float().sum()
        fn = ((true_labels == i) & (pred_labels != i)).float().sum()
        tp_and_fp = tp + fp
        tp_and_fn = tp + fn
        if tp_and_fp > 0:
            precision.append(tp / tp_and_fp)
        else:
            precision.append(0.)
        if tp_and_fn > 0:
            recall.append(tp / tp_and_fn)
        else:
            recall.append(0.)

    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    
    prec_and_recall = precision + recall
    prec_and_recall.masked_fill_(prec_and_recall == 0, 1.)
    f1 = 2 * precision * recall / prec_and_recall

    macro_avg_prec = precision.mean()
    macro_avg_recall = recall.mean()
    macro_avg_f1 = f1.mean()

    result = metric.compute()
    for i, label in vocab.enumerate():
        assert precision[i] == result["precision"][label]
        assert recall[i] == result["recall"][label]
        assert f1[i] == result["f-measure"][label]
    
    assert result["precision"]["macro avg."] == macro_avg_prec
    assert result["recall"]["macro avg."] == macro_avg_recall
    assert result["f-measure"]["macro avg."] == macro_avg_f1


