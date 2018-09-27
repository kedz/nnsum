import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNSentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, 
                 bidirectional=True, cell="gru", num_layers=1):
        super(RNNSentenceEncoder, self).__init__()

        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))

        if cell == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif cell == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, dropout=dropout)

        self.bidirectional_ = bidirectional

        if bidirectional:
            self.size_ = hidden_size * 2
        else:
            self.size_ = hidden_size

        self.dropout_ = dropout

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
