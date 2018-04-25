import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceCNNEncoder(nn.Module):
    def __init__(self, embedding_size, feature_maps=[50, 50, 100],
                 filter_windows=[1, 2, 3], dropout=0.0):
        super(SentenceCNNEncoder, self).__init__()
        self.dropout_ = dropout
        self.filters = nn.ModuleList(
            [nn.Conv3d(1, fm, (1, fw, embedding_size))
             for fm, fw in zip(feature_maps, filter_windows)])

        self.output_size_ = sum(feature_maps)

    @property
    def size(self):
        return self.output_size_

    def forward(self, inputs, word_count, input_data):
        inputs = inputs.unsqueeze(1)
        feature_maps = []
        for filter in self.filters:
            pre_act = filter(inputs).squeeze(-1)
            act = F.relu(F.max_pool2d(pre_act, (1, pre_act.size(3))))
            feature_maps.append(act.squeeze(-1))
        feature_maps = torch.cat(feature_maps, 1).permute(0, 2, 1)
        feature_maps = F.dropout(
            feature_maps, p=self.dropout_, training=self.training)
        return feature_maps

    @property
    def needs_sorted_sentences(self):
        return False
