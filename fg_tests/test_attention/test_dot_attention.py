import pytest
import torch
import torch.nn as nn
from nnsum.attention import DotAttention 


def test_singleton_matches_batch(query, context, hidden_size, batch_size):
    module = DotAttention(hidden_size, hidden_size)
    batch_attn, batch_comp = module(context, query)
    
    singleton_attns = []
    singleton_comps = []
    for b in range(batch_size):
        attn, comp = module(context[b:b+1], query[:,b:b+1,:])
        singleton_attns.append(attn)
        singleton_comps.append(comp)
    singleton_attns = torch.cat(singleton_attns, 1)
    singleton_comps = torch.cat(singleton_comps, 1)
    assert torch.allclose(singleton_attns, batch_attn)
    assert torch.allclose(singleton_comps, batch_comp)

def test_sequential_matches_batch(query, context, hidden_size, query_len):
    module = DotAttention(hidden_size, hidden_size)
    batch_attn, batch_comp = module(context, query)
    
    seq_attns = []
    seq_comps = []
    for t in range(query_len):
        attn, comp = module(context, query[t:t+1,:,:])
        seq_attns.append(attn)
        seq_comps.append(comp)
    seq_attns = torch.cat(seq_attns, 0)
    seq_comps = torch.cat(seq_comps, 0)

    assert torch.allclose(seq_attns, batch_attn, atol=1e-5)
    assert torch.allclose(seq_comps, batch_comp, atol=1e-5)

def test_mask(query, context, hidden_size, batch_size, context_len, query_len):

    ctx_mask = torch.ByteTensor(batch_size, context_len).fill_(0)
    for i in range(batch_size):
        ctx_mask[i, i:].fill_(1)

    module = DotAttention(hidden_size, hidden_size, compute_composition=False)
    attention = module(context, query, context_mask=ctx_mask)
    
    assert torch.all(torch.isnan(attention[:, 0]))
    for i in range(1, batch_size):
        assert torch.allclose(attention[:, i,:i].sum(1), 
                              torch.ones(query_len))
        assert torch.allclose(attention[:, i,i:].sum(1), 
                              torch.zeros(query_len))

def test_gradient_attn_to_query(query, context, hidden_size, query_len, 
                                batch_size):

    context_mask = torch.ByteTensor(context.size()[:2]).zero_()
    for i in range(batch_size):
        context_mask[i,i] = 1
    
    param = nn.Parameter(query)
    module = DotAttention(hidden_size, hidden_size, compute_composition=False)
    optim = torch.optim.SGD([param], lr=1.0)
    attn = module(context, param, context_mask=context_mask)
    attn = attn.contiguous().view(-1)
    context_mask = context_mask.unsqueeze(0).repeat(query_len, 1, 1).view(-1)

    for m, o in zip(context_mask, attn):
        optim.zero_grad()
        assert param.grad is None or torch.all(param.grad.eq(0))
        o.backward(retain_graph=True)
        if m.item(): 
            assert torch.all(param.grad.eq(0))
        else:
            assert torch.any(param.grad.ne(0))
        assert torch.all(~torch.isnan(param.grad))
        
def test_gradient_attn_to_context(query, context, hidden_size, query_len, 
                                  batch_size):

    context_mask = torch.ByteTensor(context.size()[:2]).zero_()
    for i in range(batch_size):
        context_mask[i,i] = 1
    
    param = nn.Parameter(context)
    module = DotAttention(hidden_size, hidden_size, compute_composition=False)
    optim = torch.optim.SGD([param], lr=1.0)
    attn = module(param, query, context_mask=context_mask)
    attn = attn.contiguous().view(-1)
    context_mask = context_mask.unsqueeze(0).repeat(query_len, 1, 1).view(-1)

    for m, o in zip(context_mask, attn):
        optim.zero_grad()
        assert param.grad is None or torch.all(param.grad.eq(0))
        o.backward(retain_graph=True)
        if m.item(): 
            assert torch.all(param.grad.eq(0))
        else:
            assert torch.any(param.grad.ne(0))
        assert torch.all(~torch.isnan(param.grad))

def test_gradient_comp_to_query(query, context, hidden_size, query_len, 
                                batch_size):

    context_mask = torch.ByteTensor(context.size()[:2]).zero_()
    for i in range(batch_size):
        context_mask[i,i] = 1
    
    param = nn.Parameter(query)
    module = DotAttention(hidden_size, hidden_size)
    optim = torch.optim.SGD([param], lr=1.0)
    attn, comp = module(context, param, context_mask=context_mask)
    comp = comp.contiguous().view(-1)

    for o in comp:
        optim.zero_grad()
        assert param.grad is None or torch.all(param.grad.eq(0))
        o.backward(retain_graph=True)
        assert torch.any(param.grad.ne(0))
        assert torch.all(~torch.isnan(param.grad))

def test_gradient_comp_to_context(query, context, hidden_size, query_len, 
                                  batch_size):

    context_mask = torch.ByteTensor(context.size()[:2]).zero_()
    for i in range(batch_size):
        context_mask[i,i] = 1
    
    param = nn.Parameter(context)
    module = DotAttention(hidden_size, hidden_size)
    optim = torch.optim.SGD([param], lr=1.0)
    attn, comp = module(param, query, context_mask=context_mask)
    comp = comp.contiguous().view(-1)

    for o in comp:
        optim.zero_grad()
        assert param.grad is None or torch.all(param.grad.eq(0))
        o.backward(retain_graph=True)
        assert torch.any(param.grad.ne(0))
        assert torch.all(~torch.isnan(param.grad))
 

#from nnsum.seq2seq import DotAttention
#from nnsum.seq2seq import RNNEncoder, RNNDecoder, EncoderDecoderModel
#from nnsum.embedding_context import EmbeddingContext
#
#
#def test_dot_attention_batch():
#    batch_size = 4
#    context_len = 10
#    hidden_size = 2
#    query_len = 6
#    context = torch.Tensor(batch_size, context_len, hidden_size).normal_()
#    query = torch.Tensor(query_len, batch_size, hidden_size).normal_()
#
#    module = DotAttention()
#    act_out, act_attn = module(context, query)
#
#    for step in range(query_len):
#        scores = torch.bmm(
#            context, query.permute(1, 2, 0)[:,:,step:step+1])
#        attention = torch.softmax(scores, dim=1)
#        read = attention.permute(0, 2, 1).bmm(context).permute(1, 0, 2)
#        ref_out = torch.cat([read, query[step:step+1]], 2)
#        assert torch.allclose(act_out[step:step+1], ref_out)
#        assert torch.allclose(act_attn[step], attention.squeeze(2))
#
#def test_dot_attention_sequential():
#    batch_size = 4
#    context_len = 10
#    hidden_size = 2
#    query_len = 6
#    context = torch.Tensor(batch_size, context_len, hidden_size).normal_()
#    query = torch.Tensor(query_len, batch_size, hidden_size).normal_()
#
#    module = DotAttention()
#
#    for step in range(query_len):
#        scores = torch.bmm(
#            context, query.permute(1, 2, 0)[:,:,step:step+1])
#        attention = torch.softmax(scores, dim=1)
#        read = attention.permute(0, 2, 1).bmm(context).permute(1, 0, 2)
#        ref_out = torch.cat([read, query[step:step+1]], 2)
#    
#        act_out, act_attn = module(context, query[step:step+1])
#        assert torch.allclose(act_out, ref_out)
#        assert torch.allclose(act_attn, attention.permute(2, 0, 1))
#

#def test_seq2seq_integration():
#    vcb_size = 10
#    hidden_dim = 32
#    src_lengths = torch.LongTensor([5, 4, 3])
#    src_features = torch.LongTensor(3, 5).random_(1, vcb_size)
#    for i in range(1, 3):
#        src_features[i,src_lengths[i]:].fill_(0)
#    src_mask = src_features.eq(0)
#    tgt_features = torch.LongTensor(3, 2).random_(3, 3 + vcb_size)
#    
#    src_ec = EmbeddingContext.from_vocab_size(
#        vcb_size, embedding_size=hidden_dim,
#        pad="<P>", start="<START>") 
#    encoder = RNNEncoder(src_ec, hidden_dim=hidden_dim)
#
#    tgt_ec = EmbeddingContext.from_vocab_size(
#        vcb_size, embedding_size=32,
#        pad="<P>", start="<START>", stop="<STOP>") 
#    decoder = RNNDecoder(tgt_ec, hidden_dim=hidden_dim, attention="dot")
#    model = EncoderDecoderModel(encoder, decoder)
#
#    logits, attn = model({"source_features": src_features,
#                          "source_lengths": src_lengths,
#                          "source_mask": src_mask,
#                          "multi_ref": False,
#                          "target_input_features": tgt_features},
#                         return_attention=True)
#    attn = attn["attention"]
#    assert torch.all(attn.masked_select(src_mask.unsqueeze(0)).eq(0))
#    for i in range(3):
#        assert torch.allclose(attn[:,i,:src_lengths[i]].sum(1), torch.ones(2))
