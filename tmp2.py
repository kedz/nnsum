import torch

x = torch.randn(10)
y, I = torch.sort(x, descending=True)
_, J = torch.sort(I)

print(x)
print(I)
print(y[J])
print((y[J] - x).abs().max())
