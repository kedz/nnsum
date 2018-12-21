import torch
from nnsum.embedding_context import EmbeddingContext
import numpy as np


def test_learning():

    vsize = 11
    emb_size = 4
    ec = EmbeddingContext.from_vocab_size(
        vsize, embedding_size=emb_size,
        pad="PAD", start="START", stop="STOP", unknown="UNK",
        transpose=False)
    
    assert len(ec.vocab) == vsize + 4
    assert ec.vocab.pad_token == "PAD"
    assert ec.vocab.unknown_token == "UNK"
    assert ec.vocab.start_token == "START"
    assert ec.vocab.stop_token == "STOP"

    # Before: orig and new should be equal. This is a sanity check.
    orig_parameters = [p.detach().clone() for p in ec.parameters()][0]
    new_parameters = [p for p in ec.parameters()][0]
    assert torch.all(orig_parameters == new_parameters)

    optimizer = torch.optim.SGD(ec.parameters(), lr=1.0)
    optimizer.zero_grad()
    targets = torch.FloatTensor(3, 5).random_()
    inputs = torch.arange(0, len(ec.vocab)).view(3, 5)
    embeddings = ec(inputs)
    mse = ((targets - embeddings.sum(2)) ** 2).mean()
    mse.backward()
    optimizer.step()

    # After: orig and new should not be equal, except for the pad index (0)
    # which should remain a zero vector. 
    assert torch.all(orig_parameters[0] == new_parameters[0])
    assert torch.all(orig_parameters[1:] != new_parameters[1:])
    assert torch.all(new_parameters[0].eq(0.))

def test_embedding_dropout():

    vsize = 15
    emb_size = 100
    emb_dropout = .3
    ec = EmbeddingContext.from_vocab_size(
        vsize, embedding_size=emb_size, embedding_dropout=emb_dropout,
        transpose=False, pad=None)
    
    assert len(ec.vocab) == vsize

    inputs = torch.arange(0, len(ec.vocab)).view(3, 5)
    
    ec.train()
    do_embeddings = ec(inputs)
    ec.eval()
    og_embeddings = ec(inputs)

    mask = do_embeddings.eq(0)
    scale = 1 / (1 - emb_dropout)

    ref = og_embeddings.data.masked_fill(mask, 0).mul(scale)
    assert torch.all(og_embeddings != do_embeddings)
    assert torch.allclose(ref, do_embeddings)

def test_token_dropout_zero_mode():
 
    vsize = 15
    emb_size = 4
    tok_dropout = .5
    ec = EmbeddingContext.from_vocab_size(
        vsize, embedding_size=emb_size, token_dropout=tok_dropout,
        transpose=False, pad=None, unknown=None)
    
    assert len(ec.vocab) == vsize

    inputs = torch.arange(0, len(ec.vocab)).view(3, 5)
    # orig and new should be equal. This is a sanity check.
    orig_parameters = [p.detach().clone() for p in ec.parameters()][0]
    new_parameters = [p for p in ec.parameters()][0]
    assert torch.all(orig_parameters == new_parameters)

    ec.train() 
    optimizer = torch.optim.SGD(ec.parameters(), lr=.01)
    optimizer.zero_grad()
    targets = torch.FloatTensor(3, 5).random_()
    inputs = torch.arange(0, len(ec.vocab)).view(3, 5)
    embeddings = ec(inputs)
    mse = ((targets - embeddings.sum(2)) ** 2).mean()
    mse.backward()
    optimizer.step()

    num_zeros = 0
    for i, row in enumerate(embeddings.view(3 * 5, -1)):
        if torch.all(row.eq(0.)):
            num_zeros += 1
            assert torch.all(new_parameters.grad[i].eq(0.))
            assert torch.all(new_parameters[i] == orig_parameters[i])
        else: 
            assert torch.all(new_parameters.grad[i].ne(0.))
            assert torch.all(new_parameters[i] != orig_parameters[i])
    assert num_zeros > 0

    ec.eval()
    embeddings2 = ec(inputs)
    for i, row in enumerate(embeddings2.view(3 * 5, -1)):
        assert torch.all(row.ne(0.))

def test_token_dropout_unknown_mode():
 
    vsize = 15
    emb_size = 4
    tok_dropout = .5
    ec = EmbeddingContext.from_vocab_size(
        vsize, embedding_size=emb_size, token_dropout=tok_dropout,
        token_dropout_mode="unknown",
        transpose=False, pad=None, unknown="<U>")
    
    assert len(ec.vocab) == vsize + 1

    # orig and new should be equal. This is a sanity check.
    orig_parameters = [p.detach().clone() for p in ec.parameters()][0]
    new_parameters = [p for p in ec.parameters()][0]
    assert torch.all(orig_parameters == new_parameters)

    ec.train() 
    optimizer = torch.optim.SGD(ec.parameters(), lr=.01)
    optimizer.zero_grad()
    targets = torch.FloatTensor(3, 5).random_()
    inputs = torch.arange(1, len(ec.vocab)).view(3, 5)
    embeddings = ec(inputs)
    mse = ((targets - embeddings.sum(2)) ** 2).mean()
    mse.backward()
    optimizer.step()

    num_unknowns = 0
    for i, row in enumerate(embeddings.view(3 * 5, -1), 1):
        if torch.all(row == orig_parameters[0]):
            num_unknowns += 1
            assert torch.all(new_parameters.grad[i].eq(0.))
            assert torch.all(new_parameters[i] == orig_parameters[i])
    assert num_unknowns > 0
    assert torch.all(new_parameters.grad[0].ne(0.))

    ec.eval()
    embeddings2 = ec(inputs)
    for i, row in enumerate(embeddings2.view(3 * 5, -1), 1):
        assert torch.all(row != new_parameters[0])

def test_apply_unknown_mode_token_dropout():
    vsize = 15
    emb_size = 4
    tok_dropout = .5
    ec = EmbeddingContext.from_vocab_size(
        vsize, embedding_size=emb_size, token_dropout=tok_dropout,
        token_dropout_mode="unknown",
        transpose=False, pad=None, unknown="<U>")
    
    assert len(ec.vocab) == vsize + 1

    ec.train() 
    
    inputs = torch.LongTensor(1, 1000).random_(1, len(ec.vocab))
    inputs_do = ec._apply_unknown_mode_token_dropout(inputs)
    unk_count = inputs_do.eq(0).long().sum()
    unk_per = unk_count / 1000
    assert np.fabs(unk_per - tok_dropout) < 1e-4

    ec.eval()
    inputs_no_do = ec._apply_unknown_mode_token_dropout(inputs)
    unk_count = inputs_no_do.eq(0).long().sum()
    assert unk_count == 0
