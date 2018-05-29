import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 bidirectional=False, cell="gru", dropout=0.0):

        super(DocumentRNNEncoder, self).__init__()

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

        if bidirectional:
            self.size_ = hidden_size * 2
        else:
            self.size_ = hidden_size
         
        self.dropout_ = dropout

    @property
    def size(self):
        return self.size_

    @property
    def dropout(self):
        return self.dropout_

    def forward(self, inputs, length):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            inputs, length.data.tolist(), batch_first=True)
        packed_output, encoder_state = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, encoder_state
