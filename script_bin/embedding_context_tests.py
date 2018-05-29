import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)
#python train_seq2seq_model.py 

train_inputs_path = "/storage/data/kedzie-summarization-data/duc-sds/inputs/duc-sds.inputs.train.json" 
train_labels_path = "/storage/data/kedzie-summarization-data/duc-sds/labels/duc-sds.labels.train.json"
pretrained_embeddings_path = "/storage/data/kedzie-summarization-data/glove/glove.6B.50d.txt"

import nnsum.io
import torch
def get_loss(inputs):
    return ((0 - inputs) ** 2).mean()


def test_update_all():

    embedding_context = nnsum.io.initialize_embedding_context(
            train_inputs_path, 50, at_least=3,
            top_k=None, 
            word_dropout=0.0,
            embedding_dropout=0.0,
            update_rule="update-all")
    W_orig = embedding_context.embeddings.weight.data.clone()
    
    data = nnsum.io.make_sds_dataset(
            train_inputs_path, train_labels_path, embedding_context.vocab)
    model = nn.Sequential(embedding_context, nn.Linear(50, 1))
    
    optim = torch.optim.Adam(model.parameters(), lr=.1) 
    for epoch in range(5):
        losses = []
        for batch in data.iter_batch():
            optim.zero_grad()
            y = model(batch.inputs.tokens)
            loss = get_loss(y)
            loss.backward()
            optim.step()
            losses.append(loss.data[0])
        print(np.mean(losses))
    print(W_orig[0:5,0:7])
    print(embedding_context.embeddings.weight.data[0:5,0:7])
    delta = (W_orig - embedding_context.embeddings.weight.data).abs()
    assert delta[1:].ne(0).all()
       

def test_fix_all():

    embedding_context = nnsum.io.initialize_embedding_context(
            train_inputs_path, 50, at_least=3,
            top_k=None, 
            word_dropout=0.0,
            embedding_dropout=0.0,
            update_rule="fix-all")
    W_orig = embedding_context.embeddings.weight.data.clone()
    
    data = nnsum.io.make_sds_dataset(
            train_inputs_path, train_labels_path, embedding_context.vocab)
    model = nn.Sequential(embedding_context, nn.Linear(50, 1))
    
    optim = torch.optim.Adam(model.parameters(), lr=.1) 
    for epoch in range(5):
        losses = []
        for batch in data.iter_batch():
            optim.zero_grad()
            y = model(batch.inputs.tokens)
            loss = get_loss(y)
            loss.backward()
            optim.step()
            losses.append(loss.data[0])
        print(np.mean(losses))
       
    print(W_orig[0:5,0:7])
    print(embedding_context.embeddings.weight.data[0:5,0:7])
    delta = (W_orig - embedding_context.embeddings.weight.data).abs()
    assert delta.eq(0).all()
    
    
test_update_all()
test_fix_all()
