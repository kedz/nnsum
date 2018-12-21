import torch.nn as nn


class ParallelEncoder(nn.Module):
    def __init__(self, encoders):
        super(ParallelEncoder, self).__init__()
        self._encoders = nn.ModuleList(encoders)
        
    def forward(self, inputs):
        return [enc(inputs) for enc in self._encoders]

    def attention(self):
        return [enc.attention()[0] for enc in self._encoders]
