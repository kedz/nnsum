import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


_desc_message = """
    Convolutional Sentence Encoder -- sentence embeddings are the output
    of a convolutional layer applied to a sentence's word embeddings.
    Output of each convolutional filter is concatenated and fed through a 
    relu+dropout layer.
    """

class CNNSentenceEncoder(nn.Module):
    def __init__(self, embedding_size, feature_maps=[50, 50, 100],
                 filter_windows=[1, 2, 3], dropout=0.0):
        super(CNNSentenceEncoder, self).__init__()
        self.dropout_ = dropout
        self.filters = nn.ModuleList(
            [nn.Conv3d(1, fm, (1, fw, embedding_size))
             for fm, fw in zip(feature_maps, filter_windows)])

        self.output_size_ = sum(feature_maps)

    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument(
            "--dropout", type=float, default=.25, required=False,
            help="Drop probability applied to convolutional output.")
        parser.add_argument(
            "--filter-windows", default=[1, 2, 3, 4, 5, 6], type=int, 
            nargs="+",
            help="Filter window widths, i.e. size in ngrams of each " \
                    "convolutional feature window. Number of args must be " \
                    "the same length as --feature-maps.")
        parser.add_argument(
            "--feature-maps", default=[25, 25, 50, 50, 50, 50], 
            type=int, nargs="+",
            help="Number of convolutional feature maps. " \
                 "Number of args must be the same length as " \
                 "--filter-windows.")
        return parser




    @property
    def size(self):
        return self.output_size_

    def forward(self, inputs, word_count):
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

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" CNNSentenceEncoder initialization started.")
        for name, p in self.named_parameters():
            if "weight" in name:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal_(p)    
            elif "bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant_(p, 0)    
            else:
                if logger:
                    logger.info(" {} ({}): random normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.normal_(p)    
        if logger:
            logger.info(" CNNSentenceEncoder initialization finished.")
