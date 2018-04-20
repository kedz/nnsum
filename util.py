import torch
import json
from embedding_layer import EmbeddingLayer


def read_vocab(data_path, topk=100000, at_least=3):
    wc = {}
    
    with open(data_path, "r") as fp:
        for line in fp:
            example = json.loads(line)
            for input in example["inputs"]:
                for token in input["tokens"]:
                    wc[token] = wc.get(token, 0) + 1

    tokens_counts = sorted(wc.items(), key=lambda x: x[1], reverse=True)

    print("# Unique Words:", len(wc))
    
    token_counts = [tc for tc in tokens_counts if tc[1] >= at_least][:topk]
    i2t = ["_PAD_", "_UNK_"] + [t for t, c in token_counts]
    t2i = {t: i for i, t in enumerate(i2t)}

    print("After filtering, # Unique Words:", len(t2i))

    return t2i, i2t

def read_pretrained_embeddings(path):
    embeddings = []
    idx2tok = ["_PAD_", "_UNK_"]
    tok2idx = {"_PAD_": 0, "_UNK_": 1}
    with open(path, "r") as fp:
        for line in fp:
            items = line.split()
            token = items[0]
            embedding = [float(x) for x in items[1:]]
            tok2idx[token] = len(idx2tok)
            idx2tok.append(token)
            embeddings.append(embedding)
    pad = [0] * len(embeddings[0])
    embeddings = torch.FloatTensor([pad, pad] + embeddings)

    print("Read {} {}-dimensional embeddings.".format(
        embeddings.size(0) - 2, embeddings.size(1)))
    return tok2idx, idx2tok, embeddings

def fill_embeddings(src_tok2idx, src_emb, tgt_idx2tok, tgt_emb):
    found = 0
    for i, t in enumerate(tgt_idx2tok):
        if t in src_tok2idx:
            tgt_emb[i].copy_(src_emb[src_tok2idx[t]])
            found += 1
    print("Filled {}/{} embeddings.".format(found, tgt_emb.size(0) - 2))

def initialize_vocab_and_embeddings(inputs_path, embedding_size, dropout=0.0,
                                    trainable=True, embeddings_path=None):
    if embeddings_path:
        print("Reading embeddings file {} ...".format(
            embeddings_path))
        pt_tok2idx, pt_idx2tok, pt_embeddings = read_pretrained_embeddings(
            embeddings_path) 
        
        if trainable:
            tok2idx, idx2tok = read_vocab(inputs_path)
            init_embeddings = torch.FloatTensor(len(idx2tok), embedding_size)
            init_embeddings.normal_()
            init_embeddings[0].fill_(0)
            fill_embeddings(
                pt_tok2idx, pt_embeddings, idx2tok, init_embeddings)
        
        else:
            tok2idx = pt_tok2idx
            idx2tok = pt_idx2tok
            init_embeddings = pt_embeddings

    else:
        tok2idx, idx2tok = read_vocab(inputs_path)
        init_embeddings = None

    embeddings = EmbeddingLayer(
        len(idx2tok), embedding_size, dropout=dropout, 
        initial_weights=init_embeddings, trainable=trainable)
    return tok2idx, idx2tok, embeddings


