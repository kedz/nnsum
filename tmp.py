import torch
from torch.autograd import Variable
import torch.nn as nn

bs = 2
ds = 3
ss = 5

x1 = torch.FloatTensor([[[1,2,3,0,0],[1,2,3,4,5],[0,0,0,0,0]]])
x2 = torch.FloatTensor([[[1,0,0,0,0],[1,2,0,0,0],[1,2,3,0,0]]])
X = Variable(torch.cat([x1, x2], 0))
lengths = Variable(torch.LongTensor([[3,5,0],[1,2,3]]))

print(X)
print(lengths)

X_flat = X.view(bs * ds, ss)
lengths_flat = lengths.view(-1)
print(X_flat)
print(lengths_flat)

lengths_flat_srt, lengths_flat_argsrt = torch.sort(
    lengths_flat, descending=True)
print(lengths_flat_srt)
print(lengths_flat_argsrt)

X_flat_srt = Variable(X_flat.data.new(*X_flat.size()))
X_flat_srt.scatter_(0, lengths_flat_argsrt.unsqueeze(1).repeat(1, ss), X_flat)
print(X_flat_srt)

lengths_flat_srt = lengths_flat_srt.masked_fill(lengths_flat_srt.eq(0), 1)

print(lengths_flat_srt)

packed_input = nn.utils.rnn.pack_padded_sequence(
    X_flat_srt.unsqueeze(2), lengths_flat_srt.data.tolist(), batch_first=True)

rnn = nn.GRU(1, 1, bidirectional=True)
output_packed, state = rnn(packed_input)

output, _ = nn.utils.rnn.pad_packed_sequence(
    output_packed, batch_first=True)
print(output)
state = state.permute(1, 0, 2)
print(state)

print(lengths_flat_argsrt)
_, lengths_flat_argsrt_inv = torch.sort(
    lengths_flat_argsrt) #, descendingA)
print(lengths_flat_argsrt_inv)

print(output[lengths_flat_argsrt_inv].view(bs, ds, ss, 2))
print(state[lengths_flat_argsrt_inv].view(bs, ds, 2))
