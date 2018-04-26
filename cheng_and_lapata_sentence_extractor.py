import torch
import torch.nn as nn
import torch.nn.functional as F


class ChengAndLapataSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 cell="gru", rnn_dropout=0.0,
                 mlp_layers=[100], mlp_dropouts=[.25]):

        super(ChengAndLapataSentenceExtractor, self).__init__()
        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))
        if cell == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout)
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout)
        else:
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout)
        self.decoder_start = nn.Parameter(
            torch.FloatTensor(input_size).normal_())

        self.teacher_forcing = True

        inp_size = hidden_size * 2
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

        if self.training and self.teacher_forcing:

            start_emb = self.decoder_start.view(1, 1, -1).repeat(bs, 1, 1)
            decoder_inputs = torch.cat(
                [start_emb, 
                 sentence_emb[:,:-1] * targets.view(bs,-1,1)[:,:-1]], 1)
    
            packed_input = nn.utils.rnn.pack_padded_sequence(
                decoder_inputs, length.data.tolist(), batch_first=True)
            packed_output, _ = self.rnn(packed_input, encoder_state)
            decoder_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)
            mlp_input = torch.cat([encoder_output, decoder_output], 2)
            logits = self.mlp(mlp_input.view(bs * ss, -1)).view(bs, ss)
            return logits

        else:
            logits = []

            sentence_emb = sentence_emb.permute(1, 0, 2)
            start_emb = self.decoder_start.view(1, 1, -1).repeat(1, bs, 1)
            state = encoder_state
            output, state = self.rnn(start_emb, state)

            logits_step = self.mlp(
                torch.cat([encoder_output[:,0], output[0]], 1))
            logits.append(logits_step)
            prev_probs = F.sigmoid(logits_step).unsqueeze(0)
            for step in range(1, ss):
                output, state = self.rnn(
                    prev_probs * sentence_emb[step - 1:step], state)
                logits_step = self.mlp(
                    torch.cat([encoder_output[:,step], output[0]], 1))
                prev_probs = F.sigmoid(logits_step).unsqueeze(0)
                logits.append(logits_step) 

            return torch.cat(logits, 1)
