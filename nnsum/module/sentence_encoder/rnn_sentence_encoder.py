import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class RNNSentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, 
                 bidirectional=True, cell="gru", num_layers=1):
        super(RNNSentenceEncoder, self).__init__()

        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))

        if cell == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, 
                              dropout=dropout if num_layers > 1 else 0.)
        elif cell == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, 
                               dropout=dropout if num_layers > 1 else 0.)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.)

        self.bidirectional_ = bidirectional

        if bidirectional:
            self.size_ = hidden_size * 2
        else:
            self.size_ = hidden_size

        self.dropout_ = dropout

    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument(
            "--hidden-size", default=300, type=int)
        parser.add_argument(
            "--bidirectional", action="store_true", default=False)
        parser.add_argument(
            "--dropout", default=.25, type=float)
        parser.add_argument(
            "--num-layers", default=1, type=int)
        parser.add_argument("--cell", choices=["rnn", "gru", "lstm"],
                            default="gru", type=str)
        return parser

    @property
    def size(self):
        return self.size_
 
    @property
    def needs_sorted_sentences(self):
        return True

    @property
    def dropout(self):
        return self.dropout_
    
    @property
    def bidirectional(self):
        return self.bidirectional_

    def forward(self, inputs, word_count):

        packed_input = nn.utils.rnn.pack_padded_sequence(
            inputs, word_count, batch_first=True)
        packed_output, encoder_state = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        if isinstance(encoder_state, tuple):
            encoder_state = encoder_state[0]

        if self.bidirectional:
            encoder_state = encoder_state[-2:].permute(1, 0, 2).contiguous()
            bs = encoder_state.size(0)
            encoder_state = encoder_state.view(bs, -1)
        else:
            encoder_state = encoder_state[-1:].permute(1, 0, 2).contiguous()
            bs = encoder_state.size(0)
            encoder_state = encoder_state.view(bs, -1)

        encoder_state = F.dropout(
            encoder_state, p=self.dropout, training=self.training)

        return encoder_state

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" RNNSentenceEncoder initialization started.")
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
            logger.info(" RNNSentenceEncoder initialization finished.")
