import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSentenceExtractor(nn.Module):
    def __init__(self, input_size, mlp_layers=[100], mlp_dropouts=[.25]):
        super(SimpleSentenceExtractor, self).__init__()

        mlp = []
        for out_size, dropout in zip(mlp_layers, mlp_dropouts):
            mlp.append(nn.Linear(input_size, out_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout))
            input_size = out_size 
        mlp.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, sentence_emb, length, encoder_output, encoder_state,
                targets=None):
        return self.mlp(encoder_output).squeeze(2)
