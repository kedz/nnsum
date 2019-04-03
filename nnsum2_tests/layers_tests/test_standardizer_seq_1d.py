import pytest
import torch
from nnsum2.layers import StandardizerSeq1D
import math


def pname(name):
    def namer(value):
        return "{}={}".format(name, str(value))
    return namer

@pytest.mark.parametrize("unbiased", [False, True], ids=pname("unbiased"))
@pytest.mark.parametrize("batch_first", [True, False], 
                         ids=pname("batch_first"))
def test_masked_input(unbiased, batch_first):
    batch_size = 3
    lengths = [3,5,4]
    emb_size = 4
    seq_size = 5

    hyperparameters = {
        "unbiased": unbiased,
        "batch_first": batch_first
    }

    if batch_first:
        inputs = torch.FloatTensor(batch_size, seq_size, emb_size).normal_()
        inputs_mask = torch.ByteTensor(batch_size, seq_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[i,length:].fill_(1)
    else:
        inputs = torch.FloatTensor(seq_size, batch_size, emb_size).normal_()
        inputs_mask = torch.ByteTensor(seq_size, batch_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[length:,i].fill_(1)

    module = StandardizerSeq1D(**hyperparameters)
    outputs, _ = module(inputs, inputs_mask=inputs_mask)
    
    singleton_outputs = []
    for i, length in enumerate(lengths):
        if batch_first:
            inputs_i = inputs[i:i+1, :length]
            time_dim = 1
            batch_dim = 0
        else:
            inputs_i = inputs[:length, i:i+1]
            time_dim = 0
            batch_dim = 1
        centered = inputs_i - inputs_i.mean(time_dim, keepdim=True)
        standard = centered / inputs_i.std(time_dim, keepdim=True, 
                                           unbiased=unbiased)
        diff = max(lengths) - length 
        if diff > 0:
            if batch_first:
                standard = torch.cat(
                    [standard,
                     torch.FloatTensor(1, diff, emb_size).fill_(0)],
                    1)
            else:
                standard = torch.cat(
                    [standard,
                     torch.FloatTensor(diff, 1, emb_size).fill_(0)],
                    0)
        singleton_outputs.append(standard)
       
    singleton_outputs = torch.cat(singleton_outputs, batch_dim)
        
    assert torch.allclose(singleton_outputs, outputs, atol=1e-6, rtol=0)
