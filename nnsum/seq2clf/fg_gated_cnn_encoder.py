import torch
import torch.nn as nn
import torch.nn.functional as F


class FGGatedCNNEncoder(nn.Module):
    def __init__(self, embedding_context, gated_window_size=7,
                 dropout=0.25):
        super(FGGatedCNNEncoder, self).__init__()
        assert (gated_window_size // 2) * 2 != gated_window_size 
        pad_size = gated_window_size // 2

        

        self._dropout = dropout
        self._embedding_context = embedding_context
        self._gate_convs = nn.Conv2d(
            1, 512, (gated_window_size, embedding_context.output_size),
            padding=(pad_size, 0))
           # padding=padding(ws))
           #  for ws in window_sizes])
        
        self._gate_ff = nn.Linear(512, 1)
        self._clf_convs = nn.ModuleList(
            nn.Conv2d(1, 100, (ws, embedding_context.output_size))
            for ws in [1, 2, 3, 4, 5])

    @property
    def output_size(self):
        return 500

    @property
    def embedding_context(self):
        return self._embedding_context

    def gate_network(self, inputs):
        inputs = inputs.unsqueeze(1)
        h1 = torch.relu(
            F.dropout(self._gate_convs(inputs).squeeze(3).permute(0, 2, 1),
                      p=self._dropout, training=self.training, inplace=True))
        return torch.sigmoid(self._gate_ff(h1))

    def forward(self, inputs, lengths):

        emb = self._embedding_context(inputs)
        #print()
        #print(emb.size())
        #gated_emb = emb.unsqueeze(1)
        gate = self.gate_network(emb)
        gated_emb = (gate * emb).unsqueeze(1)
        
        features = []
        for fltr in self._clf_convs:
            preact = F.dropout(fltr(gated_emb), p=self._dropout, 
                               training=self.training, inplace=True)
            
            #print(preact.size())

            feature = torch.relu(preact.squeeze(3).max(2)[0])
            features.append(feature)
        #input()
        features = torch.cat(features, dim=1)
        return features, {"gates": gate}

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if name == "_gate_ff.bias":
                nn.init.constant_(param, -1.)    
                print(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)    
            else:
                nn.init.normal_(param)    
