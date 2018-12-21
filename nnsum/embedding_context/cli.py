import argparse
from collections import OrderedDict
import pathlib
import torch

from .helper import create_vocab, create_label_vocab
from .embedding_context import EmbeddingContext
from .multi_embedding_context import MultiEmbeddingContext
from .label_embedding_context import LabelEmbeddingContext
from .multi_label_embedding_context import MultiLabelEmbeddingContext


def new_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, nargs="+", default=["tokens"])
    parser.add_argument("--dims", type=int, nargs="+", default=[300])
    parser.add_argument("--vocab-path", type=pathlib.Path, default=None,
                        required=False)
    parser.add_argument(
        "--token-dropout", type=float, nargs="+", default=[0.])
    parser.add_argument(
        "--token-dropout-mode", type=str, nargs="+", default=["zero"],
        choices=["zero", "unknown"])
    parser.add_argument(
        "--embedding-dropout", type=float, nargs="+", default=[None])
    
    return parser 

def new_label_context_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, nargs="+", default=["labels"])
    parser.add_argument("--vocab-path", type=pathlib.Path, default=None,
                        required=False)
    return parser 

def from_args(args, dataset, pad_token=None, unknown_token=None, 
              start_token=None, stop_token=None, features=None,
              transpose=True):

    # Validate argument lengths.
    if features is None: # TODO: think about dropping this line.
        features = args.features

    if len(features) > 1:
        if len(args.dims) == 1: 
            args.dims = args.dims * len(features)
        if len(args.embedding_dropout) == 1: 
            args.embedding_dropout = args.embedding_dropout * len(features)
        if len(args.token_dropout):
            args.token_dropout = args.token_dropout * len(features)
        if len(args.token_dropout_mode) == 1:
            args.token_dropout_mode = args.token_dropout_mode * len(features)

    if len(features) != len(args.dims):
        raise Exception(
            "--dims must have one argument or same number as features.")
    
    if len(features) != len(args.embedding_dropout):
        raise Exception(
            "--embedding-dropout must have one argument or same number as " \
            "features.")
  
    if len(features) != len(args.token_dropout):
        raise Exception(
            "--token-dropout must have one argument or same number as " \
            "features.")

    if len(features) != len(args.token_dropout_mode):
        raise Exception(
            "--token-dropout-mode must have one argument or same number as " \
            "features.")

    if args.vocab_path:
        preset_vocabs = torch.load(args.vocab_path)
    else:
        preset_vocabs = {}

    features_needing_vocab = [f for f in features if f not in preset_vocabs]

    vocabs = create_vocab(dataset, features_needing_vocab, 
                          pad_token=pad_token, 
                          unknown_token=unknown_token, start_token=start_token,
                          stop_token=stop_token)
    vocabs.update(preset_vocabs)

    print("Initializing embedding contexts:")
    contexts = []
    for feature, dims, edo, tkdo, tkdom in zip(
                features, args.dims, args.embedding_dropout,
                args.token_dropout, args.token_dropout_mode):
        vocab = vocabs[feature]

        print(" Feature: {}".format(feature))
        msg = " Dim: {:5d}  Emb. Dropout: {:5.3f}  Tkn. Dropout: {:5.3f} " \
              " Tkn. Dropout Mode: {:8s}  Vocab Size: {:7d}"
        print(msg.format(dims, edo if edo is not None else 0., 
                         tkdo, tkdom, len(vocab)))
            
        contexts.append(
            EmbeddingContext(vocab, dims, name=feature, embedding_dropout=edo,
                             token_dropout=tkdo, token_dropout_mode=tkdom,
                             transpose=transpose))

    if len(contexts) > 1:
        return MultiEmbeddingContext(contexts)
    else:
        return contexts[0]

def label_context_from_args(args, dataset, embedding_size, features=None):
    
    if features is None:
        features = args.features

    if args.vocab_path:
        preset_vocabs = torch.load(args.vocab_path)
    else:
        preset_vocabs = {}

    features_needing_vocab = [f for f in features if f not in preset_vocabs]

    vocabs = create_label_vocab(dataset, features_needing_vocab)
    vocabs.update(preset_vocabs)

    print("Initializing label embedding contexts:")
    contexts = []
    for feature, vocab in vocabs.items():
        print(" Feature: {}".format(feature))
        msg = " Dim: {:5d}  Vocab Size: {:7d}"
        print(msg.format(embedding_size, len(vocab)))
 
        contexts.append(
            LabelEmbeddingContext(vocab, embedding_size, name=feature))

    return MultiLabelEmbeddingContext(contexts)
