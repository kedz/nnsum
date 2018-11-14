import ujson as json
from collections import defaultdict, OrderedDict
from nnsum.io import Vocab


def create_vocab(dataset, features=None, start_token=None, stop_token=None,
                 pad_token=None, unk_token=None):

    if features is not None:
        feature_counts = OrderedDict()
        for feature in features:
            feature_counts[feature] = defaultdict(int)
    else:
        feature_counts = OrderedDict()
        features = []

    token_counts = defaultdict(int)

    for item in dataset:
        for token in item["tokens"]:
            token_counts[token] += 1
        for f, fc in feature_counts.items():
            for token in item[f]:
                fc[token] += 1

    token_list = sorted(token_counts, key=token_counts.get, reverse=True)
    token_vocab = Vocab.from_word_list(
        token_list, start=start_token, stop=stop_token, pad=pad_token, 
        unk=unk_token)

    feature_vocabs = OrderedDict()
    for f, fc in feature_counts.items():
        feat_list = sorted(fc, key=fc.get, reverse=True)
        feat_vocab = Vocab.from_word_list(
            feat_list, start=start_token, stop=stop_token, pad=pad_token, 
            unk=unk_token)
        feature_vocabs[f] = feat_vocab

    if len(feature_vocabs) > 0:
        return token_vocab, feature_vocabs
    else:
        return token_vocab
