import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCNNEncoder(nn.Module):
    def __init__(self, input_size, feature_maps=[50, 50, 50, 25, 25],
                 window_sizes=[1, 2, 3, 4, 5], dropout=0.0):

        assert len(feature_maps) == len(window_sizes)
        super(GatedCNNEncoder, self).__init__()

        self._dropout = dropout

        def padding(ws):
            if ws == 1:
                return (0, 0)
            elif ws == 2:
                return (1, 0)
            if ws % 2 == 0:
                return (ws // 2, 0)
            else:
                return (ws // 2, 0)

        self._filters = nn.ModuleList(
            [nn.Conv2d(1, fm, (ws, input_size), padding=padding(ws))
             for fm, ws in zip(feature_maps, window_sizes)])
        self._gates = nn.ModuleList(
            [nn.Conv2d(1, 1, (ws, input_size), padding=padding(ws))
             for ws in window_sizes])

        self._output_size = sum(feature_maps)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        feature_maps = []

        self._attention = []
        for fltr, gate in zip(self._filters, self._gates):
            preactivation = fltr(inputs).squeeze(3)
            gates = gate(inputs).squeeze(3)
            #if mask is not None:
            #    from warnings import warn
            #    warn("Mask is going to break at some point. you've been warned")
            #    gates.data.masked_fill_(mask.unsqueeze(1), float("-inf"))
            gates = torch.sigmoid(gates)
            #print(gates)
            #print(gates.size())
            #print(mask.unsqueeze(1).size())

            #input()
            self._attention.append(gates.squeeze(1).detach())
            
            preactivation = gates * preactivation
            act = F.tanh(
                F.max_pool2d(preactivation, (1, preactivation.size(2))))
            feature_maps.append(act.squeeze(2))

        feature_maps = torch.cat(feature_maps, 1)
        feature_maps = F.dropout(
            feature_maps, p=self._dropout, training=self.training)

        return feature_maps

    def attention(self):
        return self._attention
