from .vocab import Vocab
from ..module import EmbeddingContext

import torch
import json

import logging

from multiprocessing.pool import Pool

def _process_file(path):
    wc = {}
    with path.open("r") as fp:
        for sent in json.loads(fp.read())["inputs"]:
            for token in sent["tokens"]:
                token = token.lower()
                wc[token] = wc.get(token, 0) + 1
    return wc

def create_vocab(inputs_path, top_k=1000000000, at_least=1, pad="_PAD_", 
                 unk="_UNK_", processes=32):



    word_counts = {}
    pool = Pool(processes)

    for wc in pool.imap_unordered(_process_file, inputs_path.glob("*.json")):
        for k, v in wc.items():
            word_counts[k] = word_counts.get(k, 0) + v

    tokens_counts = sorted(
        word_counts.items(), key=lambda x: x[1], reverse=True)

    logging.info(" # Unique Words: {}".format(len(word_counts)))

    sorted_tokens_counts = [tc for tc in tokens_counts 
                            if tc[1] >= at_least][:top_k]

    index2tokens = []
    if pad is not None:
        index2tokens.append(pad)
    if unk is not None:
        index2tokens.append(unk)
    index2tokens.extend([t for t, c in sorted_tokens_counts])
    tokens2index = {t: i for i, t in enumerate(index2tokens)}

    logging.info(" After filtering, # Unique Words: {}".format(
        len(tokens2index)))

    return Vocab(index2tokens, tokens2index, pad=pad, unk=unk)

def load_pretrained_embeddings(path, append_pad=None, append_unknown=None):
    index2tokens = []
    tokens2index = {}
    embeddings = []

    if append_pad is not None:
        tokens2index[append_pad] = len(index2tokens)
        index2tokens.append(append_pad)

    if append_unknown is not None:
        tokens2index[append_unknown] = len(index2tokens)
        index2tokens.append(append_unknown)

    logging.info(" Reading pretrained embeddings from {}".format(path))
    with open(path, "r") as fp:
        for line in fp:
            items = line.split()
            token = items[0]
            embedding = [float(x) for x in items[1:]]
            tokens2index[token] = len(index2tokens)
            index2tokens.append(token)
            embeddings.append(embedding)

    if append_unknown is not None:
        embeddings = [[0] * len(embeddings[-1])] + embeddings

    if append_pad is not None:
        embeddings = [[0] * len(embeddings[-1])] + embeddings

    embeddings = torch.FloatTensor(embeddings)
    vocab = Vocab(
        index2tokens, tokens2index, pad=append_pad, unk=append_unknown)
    logging.info(" Read {} embeddings of size {}".format(*embeddings.size()))

    return vocab, embeddings


def filter_embeddings(vocab, embeddings, filter_vocab):

    f_index2tokens = []
    f_tokens2index = {}
    f_embeddings = []
    for index, token in filter_vocab.enumerate():
        if token in vocab:
            f_tokens2index[token] = len(f_index2tokens)
            f_index2tokens.append(token)
            f_embeddings.append(embeddings[index:index + 1])
            logging.debug(" Found {}".format(token))
    f_embeddings = torch.cat(f_embeddings, dim=0)
   
    f_vocab = Vocab(
        f_index2tokens, f_tokens2index, pad=filter_vocab.pad_token, 
        unk=filter_vocab.unknown_token)
    logging.info(" Found {} words ({:6.2f}%).".format(
        len(f_vocab), len(f_vocab) / len(filter_vocab) * 100))

    return f_vocab, f_embeddings
 
def initialize_embedding_context(inputs_path, embedding_size=300, 
                                 update_rule="update-all", word_dropout=0.0,
                                 embedding_dropout=0.0, 
                                 pretrained_embeddings=None, at_least=1,
                                 filter_pretrained=False,
                                 pretrained_append_pad="_PAD_",
                                 pretrained_append_unknown="_UNK_",
                                 top_k=None):
    if pretrained_embeddings:
        pt_vocab, pt_embeddings = load_pretrained_embeddings(
            pretrained_embeddings, 
            append_pad=pretrained_append_pad,
            append_unknown=pretrained_append_unknown)
        if filter_pretrained:
            filter_vocab = create_vocab(
                inputs_path, at_least=at_least, top_k=top_k)
            vocab, initializer = filter_embeddings(
                pt_vocab, pt_embeddings, filter_vocab)
            
        else:
            vocab = pt_vocab
            initializer = pt_embeddings

    else:
        logging.info(" Creating new embeddings with normal initializaion.")
        vocab = create_vocab(
            inputs_path, at_least=at_least, top_k=top_k)
        if update_rule != "update-all":
            logging.warn(
                " Embeddings are randomly initialized but" \
                " update rule is not 'update-all'") 

        initializer = None 

    ec = EmbeddingContext(
        vocab, embedding_size, initializer=initializer, 
        word_dropout=word_dropout, embedding_dropout=embedding_dropout,
        update_rule=update_rule)

    logging.info(" " + repr(ec))
    return ec
