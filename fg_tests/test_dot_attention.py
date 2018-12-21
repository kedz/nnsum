import torch
from nnsum.seq2seq import DotAttention
from nnsum.seq2seq import RNNEncoder, RNNDecoder, EncoderDecoderModel
from nnsum.embedding_context import EmbeddingContext


def test_dot_attention_batch():
    batch_size = 4
    context_len = 10
    hidden_size = 2
    query_len = 6
    context = torch.Tensor(batch_size, context_len, hidden_size).normal_()
    query = torch.Tensor(query_len, batch_size, hidden_size).normal_()

    module = DotAttention()
    act_out, act_attn = module(context, query)

    for step in range(query_len):
        scores = torch.bmm(
            context, query.permute(1, 2, 0)[:,:,step:step+1])
        attention = torch.softmax(scores, dim=1)
        read = attention.permute(0, 2, 1).bmm(context).permute(1, 0, 2)
        ref_out = torch.cat([read, query[step:step+1]], 2)
        assert torch.allclose(act_out[step:step+1], ref_out)
        assert torch.allclose(act_attn[step], attention.squeeze(2))

def test_dot_attention_sequential():
    batch_size = 4
    context_len = 10
    hidden_size = 2
    query_len = 6
    context = torch.Tensor(batch_size, context_len, hidden_size).normal_()
    query = torch.Tensor(query_len, batch_size, hidden_size).normal_()

    module = DotAttention()

    for step in range(query_len):
        scores = torch.bmm(
            context, query.permute(1, 2, 0)[:,:,step:step+1])
        attention = torch.softmax(scores, dim=1)
        read = attention.permute(0, 2, 1).bmm(context).permute(1, 0, 2)
        ref_out = torch.cat([read, query[step:step+1]], 2)
    
        act_out, act_attn = module(context, query[step:step+1])
        assert torch.allclose(act_out, ref_out)
        assert torch.allclose(act_attn, attention.permute(2, 0, 1))

def test_dot_attention_mask():

    batch_size = 3
    context_len = 4
    hidden_size = 2
    query_len = 6

    ctx_mask = torch.ByteTensor(batch_size, context_len).fill_(0)
    
    context = torch.Tensor(batch_size, context_len, hidden_size).normal_()
    for i in range(batch_size):
        context[i, i:].fill_(0)
        ctx_mask[i, i:].fill_(1)

    query = torch.Tensor(query_len, batch_size, hidden_size).normal_()

    module = DotAttention()
    attention = module(context, query, mask=ctx_mask)
    assert torch.all(torch.isnan(attention[1][:, 0]))
    assert torch.all(torch.isnan(attention[0][:,0,:2]))
    assert torch.allclose(attention[0][:,0,2:], query[:, 0])
    for i in range(1, batch_size):
        assert torch.allclose(attention[1][:, i,:i].sum(1), 
                              torch.ones(query_len))
        assert torch.allclose(attention[1][:, i,i:].sum(1), 
                              torch.zeros(query_len))
        assert torch.allclose(attention[0][:,i,2:], query[:, i])

def test_seq2seq_integration():
    vcb_size = 10
    hidden_dim = 32
    src_lengths = torch.LongTensor([5, 4, 3])
    src_features = torch.LongTensor(3, 5).random_(1, vcb_size)
    for i in range(1, 3):
        src_features[i,src_lengths[i]:].fill_(0)
    src_mask = src_features.eq(0)
    tgt_features = torch.LongTensor(3, 2).random_(3, 3 + vcb_size)
    
    src_ec = EmbeddingContext.from_vocab_size(
        vcb_size, embedding_size=hidden_dim,
        pad="<P>", start="<START>") 
    encoder = RNNEncoder(src_ec, hidden_dim=hidden_dim)

    tgt_ec = EmbeddingContext.from_vocab_size(
        vcb_size, embedding_size=32,
        pad="<P>", start="<START>", stop="<STOP>") 
    decoder = RNNDecoder(tgt_ec, hidden_dim=hidden_dim, attention="dot")
    model = EncoderDecoderModel(encoder, decoder)

    logits, attn = model({"source_features": src_features,
                          "source_lengths": src_lengths,
                          "source_mask": src_mask,
                          "multi_ref": False,
                          "target_input_features": tgt_features},
                         return_attention=True)
    attn = attn["attention"]
    assert torch.all(attn.masked_select(src_mask.unsqueeze(0)).eq(0))
    for i in range(3):
        assert torch.allclose(attn[:,i,:src_lengths[i]].sum(1), torch.ones(2))
