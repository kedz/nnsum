import nnsum
from nnsum.embedding_context import EmbeddingContext
import torch
import numpy as np


def test_gru_no_attn_forward():

    batch_size = 3
    vocab_size = 10
    seq_size = 6

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32) 
    decoder = nnsum.seq2seq.RNNDecoder(emb_ctx, hidden_dim=32)

    # NO INIT STATE
    X = torch.LongTensor(batch_size, seq_size).random_(0, vocab_size)
    Y = []
    state = None
    for step in range(seq_size):
        x = X[:, step:step + 1]
        y, _, state = decoder(x, None, state)
        Y.append(y)
    Y_expected = torch.cat(Y, dim=0)   
    Y_actual, _, _ = decoder(X, None, None)
    assert torch.allclose(Y_expected, Y_actual)

    # WITH INIT STATE 
    Y = []
    start_state = torch.FloatTensor(1, batch_size, 32).normal_()
    state = start_state
    for step in range(seq_size):
        x = X[:, step:step + 1]
        y, _, state = decoder(x, None, state)
        Y.append(y)
    Y_expected = torch.cat(Y, dim=0)   
    Y_actual, _, _ = decoder(X, None, start_state)
    assert torch.allclose(Y_expected, Y_actual)

def test_gru_dot_attn_forward():
    batch_size = 3
    vocab_size = 10
    seq_size = 6
    src_size = 5
    hidden_size = 32

    src_ctx = torch.FloatTensor(batch_size, src_size, hidden_size).normal_()
    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32) 
    decoder = nnsum.seq2seq.RNNDecoder(emb_ctx, hidden_dim=hidden_size,
                                       attention="dot")

    X = torch.LongTensor(batch_size, seq_size).random_(0, vocab_size)
    Y = []
    A = []
    state = None
    for step in range(seq_size):
        x = X[:, step:step + 1]
        y, attn_state, state = decoder(x, src_ctx, state)
        Y.append(y)
        A.append(attn_state["attention"])
    Y_expected = torch.cat(Y, dim=0)   
    A_expected = torch.cat(A, dim=0)
    Y_actual, actual_attn_dict, _ = decoder(X, src_ctx, None)
    A_actual = actual_attn_dict["attention"]
    assert torch.allclose(A_expected, A_actual)
    assert torch.allclose(Y_expected, Y_actual, .001)

    # WITH INIT STATE 
    Y = []
    A = []
    start_state = torch.FloatTensor(1, batch_size, 32).normal_()
    state = start_state
    for step in range(seq_size):
        x = X[:, step:step + 1]
        y, attn_state, state = decoder(x, src_ctx, state)
        Y.append(y)
        A.append(attn_state["attention"])
    Y_expected = torch.cat(Y, dim=0)   
    A_expected = torch.cat(A, dim=0)
    Y_actual, actual_attn_dict, _ = decoder(X, src_ctx, start_state)
    A_actual = actual_attn_dict["attention"]
    assert torch.allclose(A_expected, A_actual)
    assert torch.allclose(Y_expected, Y_actual, .001)

def test_gru_no_attn_decode():

    batch_size = 3
    vocab_size = 3
    seq_size = 6
    max_steps = 25
    hidden_dim = 32

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = nnsum.seq2seq.RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    batch_decode, batch_attn, batch_log_probs = decoder.decode(
        context, init_state, max_steps=max_steps, return_log_probs=True) 
    
    inputs = torch.cat(
        [torch.LongTensor(batch_size, 1).fill_(emb_ctx.vocab.start_index),
         batch_decode[:,:-1]],
        1)
    inputs.data.masked_fill_(inputs.eq(emb_ctx.vocab.stop_index), 
                             emb_ctx.vocab.pad_index)
    forward_logits, _, _  = decoder(inputs, context, init_state)
    forward_lps = torch.log_softmax(forward_logits, dim=2)
    expected_token_lps = forward_lps.gather(2, batch_decode.t().unsqueeze(2))
    for b in range(batch_size):
        stop_step = np.where(
            batch_decode[b,:].detach().numpy() == emb_ctx.vocab.stop_index)
        if len(stop_step[0]):
            expected_token_lps.data[stop_step[0][0] + 1:, b].fill_(0)

    # Check that forward log probs match decode log probs.
    assert torch.allclose(expected_token_lps, batch_log_probs, 1e-5)

    # Check that batch version matches singleton version
    for b in range(batch_size):
        context_b = context[b:b+1]
        init_state_b = init_state[:,b:b+1]
        decode_b, attn_b, log_probs_b = decoder.decode(
            context_b, init_state_b, max_steps=max_steps, 
            return_log_probs=True) 
        out_len = decode_b.size(1)
        assert torch.all(batch_decode[b:b+1,:out_len] == decode_b)
        assert torch.allclose(batch_log_probs[:out_len,b:b+1], log_probs_b)

def test_gru_dot_attn_decode():

    batch_size = 3
    vocab_size = 10
    seq_size = 6
    max_steps = 4
    hidden_dim = 32

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = nnsum.seq2seq.RNNDecoder(emb_ctx, hidden_dim=hidden_dim,
                                       attention="dot")
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    batch_decode, batch_attn, batch_log_probs = decoder.decode(
        context, init_state, max_steps=max_steps, return_log_probs=True) 
    batch_attn = torch.cat([b["attention"] for b in batch_attn], dim=0)
    
    inputs = torch.cat(
        [torch.LongTensor(batch_size, 1).fill_(emb_ctx.vocab.start_index),
         batch_decode[:,:-1]],
        1)
    inputs.data.masked_fill_(inputs.eq(emb_ctx.vocab.stop_index), 
                             emb_ctx.vocab.pad_index)
    forward_logits, forward_attn, _  = decoder(inputs, context, init_state)
    forward_attn = forward_attn["attention"]
    forward_lps = torch.log_softmax(forward_logits, dim=2)
    expected_token_lps = forward_lps.gather(2, batch_decode.t().unsqueeze(2))
    for b in range(batch_size):
        stop_step = np.where(
            batch_decode[b,:].detach().numpy() == emb_ctx.vocab.stop_index)
        if len(stop_step[0]):
            expected_token_lps.data[stop_step[0][0] + 1:, b].fill_(0)
            forward_attn[stop_step[0][0] + 1:, b].fill_(0)
            batch_attn[stop_step[0][0] + 1:, b].fill_(0)
    
    # Check that forward log probs match decode log probs.
    assert torch.allclose(expected_token_lps, batch_log_probs, 1e-5)

    # Check that forward attention matches decode attention.
    assert torch.allclose(batch_attn, forward_attn)

    # Check that batch version matches singleton version
    for b in range(batch_size):
        context_b = context[b:b+1]
        init_state_b = init_state[:,b:b+1]
        decode_b, attn_b, log_probs_b = decoder.decode(
            context_b, init_state_b, max_steps=max_steps, 
            return_log_probs=True) 
        attn_b = torch.cat([a["attention"] for a in attn_b], dim=0)
        out_len = decode_b.size(1)
        assert torch.all(batch_decode[b:b+1,:out_len] == decode_b)
        assert torch.allclose(batch_log_probs[:out_len,b:b+1], log_probs_b)
        assert torch.allclose(batch_attn[:out_len,b:b+1], attn_b)
