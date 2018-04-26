import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import NoAttention, DotAttention


class RNNSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 cell="gru", rnn_dropout=0.0, bidirectional=False,
                 mlp_layers=[100], mlp_dropouts=[.25],
                 attention="dot"):

        super(RNNSentenceExtractor, self).__init__()
        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))
        if cell == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout, bidirectional=bidirectional)
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout, bidirectional=bidirectional)
        self.decoder_start = nn.Parameter(
            torch.FloatTensor(input_size).normal_())

        if attention == "dot":
            self.attention = DotAttention()
        elif attention is None:
            self.attention = NoAttention()
        else:
            raise Exception("attention must be None or 'dot'.")

        self.teacher_forcing = True

        inp_size = hidden_size
        if bidirectional:
            inp_size *= 2

        if attention is not None:
            inp_size *= 2
        mlp = []
        for out_size, dropout in zip(mlp_layers, mlp_dropouts):
            mlp.append(nn.Linear(inp_size, out_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout))
            inp_size = out_size 
        mlp.append(nn.Linear(inp_size, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, sentence_emb, length, encoder_output, encoder_state,
                targets=None):

        bs = sentence_emb.size(0)
        ss = sentence_emb.size(1)
        start_emb = self.decoder_start.view(1, 1, -1).repeat(1, bs, 1)
        _, decoder_start_state = self.rnn(start_emb, encoder_state)

        decoder_input = nn.utils.rnn.pack_padded_sequence(
            sentence_emb, length.data.tolist(), batch_first=True)
        packed_decoder_output, _ = self.rnn(decoder_input, decoder_start_state)
        decoder_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_decoder_output, batch_first=True)

        mlp_input, scores = self.attention(
            encoder_output, decoder_output, length)

        return self.mlp(mlp_input).squeeze(2)
