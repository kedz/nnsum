import nnsum
from nnsum.embedding_context import EmbeddingContext
from nnsum.seq2seq import RNNDecoder, BeamSearch
import torch
import numpy as np


def test_gru_initialize_state():

    batch_size = 3
    vocab_size = 10
    seq_size = 6
    max_steps = 25
    hidden_dim = 32
    beam_size = 8

    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    
    class MockBeamSearch(object):
        def __init__(self):
            self._beam_size = beam_size

    beam_init_state = BeamSearch._initialize_state(
        MockBeamSearch(), init_state)
    beam_init_state = beam_init_state.view(
        1, batch_size, beam_size, hidden_dim)
        
    for b in range(beam_size):
        assert torch.all(beam_init_state[:,:,b] == init_state)

def test_initialize_context():

    batch_size = 3
    ctx_size = 32
    ctx_len = 4
    beam_size = 8

    context = torch.FloatTensor(batch_size, ctx_len, ctx_size).normal_()
    
    class MockBeamSearch(object):
        def __init__(self):
            self._beam_size = beam_size

    beam_context = BeamSearch._initialize_context(
        MockBeamSearch(), context)
    beam_context = beam_context.view(
        batch_size, beam_size, ctx_len, ctx_size)
        
    for b in range(beam_size):
        assert torch.all(beam_context[:,b] == context)

def test_gru_no_attn_beam_search_top1_matches_decode():

    batch_size = 3
    vocab_size = 10
    seq_size = 6
    max_steps = 25
    hidden_dim = 5
    beam_size = 1

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    ref_decode, _, ref_lps = decoder.decode(context, init_state, 
                                            max_steps=max_steps,
                                            return_log_probs=True)

    beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                      max_steps=max_steps)
    beam.search(return_incomplete=True)
    assert torch.all(ref_decode.eq(beam.candidates.squeeze(1)))
    assert torch.allclose(beam.log_probs, ref_lps.sum(0))

    assert torch.allclose(beam.log_probs / beam.lengths.float(), beam.scores)


def test_gru_no_attn_forward_matches_beam_search():

    batch_size = 3
    vocab_size = 10
    seq_size = 6
    max_steps = 25
    hidden_dim = 5
    beam_size = 8

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                      max_steps=max_steps)
    beam.search(return_incomplete=True)
    outputs = beam.candidates.view(batch_size * beam_size, -1)
    init_state = beam._initialize_state(init_state)
    context = beam._initialize_state(context)

    inputs = torch.cat(
        [torch.LongTensor(batch_size * beam_size, 1).fill_(1),
         outputs[:,:-1]], 1)
    inputs.data.masked_fill_(inputs.eq(2), 0)

    logits, _, _ = decoder(inputs, context, init_state)
    log_probs = torch.log_softmax(logits, 2)


    log_probs = log_probs.gather(2, outputs.unsqueeze(-1).permute(1,0,2))


    mask = outputs.t().eq(2).cumsum(0).masked_fill(outputs.t().eq(2), 0).byte()
    log_probs = log_probs.squeeze(2).masked_fill(mask, 0)
    log_probs = log_probs.sum(0).view(batch_size, beam_size)
    assert torch.allclose(log_probs, beam.log_probs)
    assert torch.allclose(beam.log_probs / beam.lengths.float(), beam.scores)

def test_gru_no_attn_no_return_incomplete():

    batch_size = 3
    vocab_size = 50
    seq_size = 3
    max_steps = 5
    hidden_dim = 5
    beam_size = 25

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                      max_steps=max_steps)
    beam.search(return_incomplete=False)

    ninf = float("-inf")
    for i in range(batch_size):
        for j in range(beam_size):
            if beam.lengths[i,j].item() == 0:
                assert beam.log_probs[i,j].item() == ninf
                assert beam.scores[i,j].item() == ninf
                assert torch.all(beam.candidates[i,j].eq(0))
            else:
                assert beam.log_probs[i,j].item() != ninf
                assert beam.scores[i,j].item() != ninf
                assert torch.allclose(
                    beam.log_probs[i,j] / beam.lengths[i,j].float(),
                    beam.scores[i,j])

def test_gru_no_attn_learns():
    batch_size = 3
    vocab_size = 10
    ctx_size = 3
    max_steps = 25
    hidden_dim = 5
    beam_size = 4

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.train()
    orig_params = [p.clone() for p in decoder.parameters()]
    updated_params = decoder.parameters()
    optimizer = torch.optim.SGD(updated_params, lr=1.0)

    init_state = torch.Tensor(1, batch_size, hidden_dim).normal_()
    context = torch.Tensor(batch_size, ctx_size, hidden_dim).normal_() 

    optimizer.zero_grad()
    beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                      max_steps=max_steps)
    beam.search(return_incomplete=True)
    loss = beam.log_probs.mul(-1).mean()
    loss.backward()
    optimizer.step()
    for mod_param, orig_param in zip(decoder.parameters(), orig_params):
        assert mod_param.size() == orig_param.size()
        assert not torch.allclose(mod_param, orig_param)


def test_gru_no_attn_sort_scores():

    batch_size = 3
    vocab_size = 10
    seq_size = 3
    max_steps = 100
    hidden_dim = 5
    beam_size = 8

    emb_ctx = EmbeddingContext.from_vocab_size(vocab_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(emb_ctx, hidden_dim=hidden_dim)
    decoder.eval()
    init_state = torch.FloatTensor(1, batch_size, hidden_dim).normal_()
    context = torch.FloatTensor(batch_size, 4, hidden_dim).normal_() 

    beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                      max_steps=max_steps)
    beam.search(return_incomplete=True)

    st_beam = BeamSearch(decoder, init_state, context, beam_size=beam_size,
                         max_steps=max_steps)
    st_beam.search(return_incomplete=True)
    st_beam.sort_by_score()

    for i in range(batch_size):
        indices = np.argsort(beam.scores[i].detach().numpy())[::-1]
        for js, j in enumerate(indices):
            assert beam.scores[i,j].item() == st_beam.scores[i,js].item()
            assert beam.log_probs[i,j] == st_beam.log_probs[i,js]
            assert beam.lengths[i,j] == st_beam.lengths[i,js]
            assert torch.all(
                beam.candidates[i,j].eq(st_beam.candidates[i,js]))

    
        

