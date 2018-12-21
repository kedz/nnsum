import ujson as json
from collections import defaultdict, OrderedDict

from .vocab import Vocab


def create_vocab(dataset, features, start_token=None, stop_token=None,
                 pad_token=None, unknown_token=None):

    feature_counts = OrderedDict()
    for feature in features:
        feature_counts[feature] = defaultdict(int)

    for item in dataset:
        if "references" in item:
            for ref in item["references"]:
                for f, fc in feature_counts.items():
                    for token in ref["tokens"][f]:
                        fc[token] += 1
        else:
            for f, fc in feature_counts.items():
                for token in item["tokens"][f]:
                    fc[token] += 1

    feature_vocabs = OrderedDict()
    for f, fc in feature_counts.items():
        feat_list = sorted(fc, key=fc.get, reverse=True)
        feat_vocab = Vocab.from_word_list(
            feat_list, start=start_token, stop=stop_token, pad=pad_token, 
            unk=unknown_token)
        feature_vocabs[f] = feat_vocab
        
    return feature_vocabs

def create_label_vocab(dataset, features):

    import warnings
    warnings.warn("Add optional n/a value argument.")
    label_counts = OrderedDict()
    for label in features:
        label_counts[label] = defaultdict(int)

    for item in dataset:
        for f, fc in label_counts.items():
            label = item["labels"].get(f, None)
            if label is None:
                label = "(n/a)"
            #if label is not None:
            fc[label] += 1

    label_vocabs = OrderedDict()
    for f, fc in label_counts.items():
        feat_list = sorted(fc, key=fc.get, reverse=True)
        feat_vocab = Vocab.from_word_list(
            feat_list, start=None, stop=None, pad=None, unk=None,
            counts=fc)
        label_vocabs[f] = feat_vocab        

    return label_vocabs
