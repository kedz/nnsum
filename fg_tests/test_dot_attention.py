import torch
from nnsum.seq2seq import DotAttention


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
