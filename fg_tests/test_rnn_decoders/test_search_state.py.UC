import torch
import torch.nn as nn
import nnsum.seq2seq as s2s


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(25, 10)
        self.hid = nn.Sequential(nn.Linear(10 + 10, 10), nn.Tanh())
        self.attn = nn.Sequential(nn.Linear(10, 5), nn.Tanh())
        self.logits = nn.Linear(10,10)

    def forward(self, inputs, prev_hid):
        emb = self.emb(inputs.t())
        hid = self.hid(torch.cat([emb, prev_hid], dim=2))
        attn = self.attn(hid)
        logits = self.logits(hid)
        return hid, attn, logits

def test_append_state():

    batch_size = 3
    seq_size = 4
    data_size = 5

    batch1 = torch.FloatTensor(1, batch_size).normal_()
    batch2 = torch.FloatTensor(1, batch_size).normal_()
    batch3 = torch.FloatTensor(1, batch_size).normal_()
    batch4 = torch.FloatTensor(1, batch_size).normal_()
    seq1 = torch.FloatTensor(1, batch_size, data_size).normal_()
    seq2 = torch.FloatTensor(1, batch_size, data_size).normal_()
    seq3 = torch.FloatTensor(2, batch_size, data_size).normal_()
    seq4 = torch.FloatTensor(4, batch_size, data_size).normal_()

    s1 = s2s.SearchState(batch=batch1, seq=seq1)
    s2 = s2s.SearchState(batch=batch2, seq=seq2)
    s1.append(s2)
    assert torch.all(
        s1["batch"] == torch.cat([batch1, batch2], dim=0))
    assert torch.all(
        s1["seq"] == torch.cat([seq1, seq2], dim=0))

    s1 = s2s.SearchState(batch=batch1, seq=seq1)
    s2 = s2s.SearchState(batch=batch2, seq=seq2)
    s3 = s2s.SearchState(batch=batch3, seq=seq3)
    s1.append(s2).append(s3)
    assert torch.all(
        s1["batch"] == torch.cat([batch1, batch2, batch3], dim=0))
    assert torch.all(
        s1["seq"] == torch.cat([seq1, seq2, seq3], dim=0))

    s1 = s2s.SearchState(batch=batch1, seq=seq1)
    s2 = s2s.SearchState(batch=batch2, seq=seq2)
    s3 = s2s.SearchState(batch=batch3, seq=seq3)
    s1.append(s2.append(s3))
    assert torch.all(
        s1["batch"] == torch.cat([batch1, batch2, batch3], dim=0))
    assert torch.all(
        s1["seq"] == torch.cat([seq1, seq2, seq3], dim=0))

    s1 = s2s.SearchState(batch=batch1, seq=seq1)
    s2 = s2s.SearchState(batch=batch2, seq=seq2)
    s3 = s2s.SearchState(batch=batch3, seq=seq3)
    s4 = s2s.SearchState(batch=batch4, seq=seq4)
    s12 = s1.append(s2)
    s34 = s3.append(s4)
    s1234 = s12.append(s34)
    assert torch.all(
        s1234["batch"] == torch.cat([batch1, batch2, batch3, batch4], dim=0))
    assert torch.all(
        s1234["seq"] == torch.cat([seq1, seq2, seq3, seq4], dim=0))



    return

def test_lazy_reindexing():


    print()
    print("HI!")
    I = torch.LongTensor([1,0,2])
    def foo(*args):
        for arg in args:
            print(type(arg))
    foo(slice(0,3), I)
    X = torch.FloatTensor(2,3).normal_()
    print(X)
    print(X[slice(0,2), I])
    #[]
    indexers = tuple([slice(0,2), I])
    print(X[indexers])
