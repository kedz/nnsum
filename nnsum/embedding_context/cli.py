import argparse
from collections import OrderedDict

from .helper import create_vocab, create_label_vocab
from .embedding_context import EmbeddingContext
from .multi_embedding_context import MultiEmbeddingContext
from .label_embedding_context import LabelEmbeddingContext
from .multi_label_embedding_context import MultiLabelEmbeddingContext


def new_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, nargs="+", default=["tokens"])
    parser.add_argument("--dims", type=int, nargs="+", default=[300])
#    parser.add_argument(
#        "--feature-dropout", type=float, nargs="+", default=None)
#    parser.add_argument(
#        "--embedding-dropout", type=float, nargs="+", default=None)
    
    return parser 

def from_args(args, dataset, pad_token=None, unknown_token=None, 
              start_token=None, stop_token=None, features=None,
              transpose=True):

    all_dims = args.dims
    if features is None:
        features = args.features
    vocabs = create_vocab(dataset, features, pad_token=pad_token, 
                          unknown_token=unknown_token, start_token=start_token,
                          stop_token=stop_token)
    contexts = []
    for (feature, vocab), dims, in zip(vocabs.items(), all_dims):
        contexts.append(EmbeddingContext(vocab, dims, name=feature, 
                                         transpose=transpose))

    if len(contexts) > 1:
        return MultiEmbeddingContext(contexts)
    else:
        return contexts[0]

def label_context_from_args(args, dataset, features=None):
    
    all_dims = args.dims
    if features is None:
        features = args.features
    vocabs = create_label_vocab(dataset, features)

    contexts = []
    for (feature, vocab), dims, in zip(vocabs.items(), all_dims):
        contexts.append(LabelEmbeddingContext(vocab, dims, name=feature))

    if len(contexts) > 1:
        return MultiLabelEmbeddingContext(contexts)
    else:
        return contexts[0]
