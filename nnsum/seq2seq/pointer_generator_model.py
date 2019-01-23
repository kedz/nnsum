import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGeneratorModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(PointerGeneratorModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def encode(self):
        pass


    def xentropy(self, batch, reduction="mean", return_attention=False):

        pass

