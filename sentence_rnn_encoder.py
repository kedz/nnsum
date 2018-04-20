import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, 
                 bidirectional=True, cell="gru"):
        super(SentenceRNNEncoder, self).__init__()

        if cell not in ["GRU", "LSTM", "RNN"]:
            raise Exception(("cell expected one of 'GRU', 'LSTM', or 'RNN' "
                             "but got {}").format(cell))

        num_layers = 1
        if cell == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif cell == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, dropout=dropout)

        if bidirectional:
            self.size_ = hidden_size * 2
        else:
            self.size_ = hidden_size

    @property
    def size(self):
        return self.size_
 
    @property
    def needs_sorted_sentences(self):
        return True


    def forward(self, inputs, input_data):

        print(inputs)
        exit()
