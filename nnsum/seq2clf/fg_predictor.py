import torch
import torch.nn as nn
import torch.nn.functional as F


class FGPredictor(nn.Module):
    def __init__(self, input_size, embedding_context, dropout=.25):
        super(FGPredictor, self).__init__()
        self._layer1 = nn.Linear(input_size, embedding_context.embedding_size)
        self._embedding_context = embedding_context
        self._dropout = dropout

    @property
    def embedding_context(self):
        return self._embedding_context

    def forward(self, inputs):
        hidden = torch.relu(
            F.dropout(self._layer1(inputs), p=self._dropout, 
                      training=self.training, inplace=True))
        return self.embedding_context(hidden)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 1.)    
            else:
                nn.init.normal_(param)    
