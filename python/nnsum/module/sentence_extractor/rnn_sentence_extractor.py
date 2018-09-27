import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 bidirectional=True, cell="gru", rnn_dropout=0.0,
                 mlp_layers=[100], mlp_dropouts=[.25]):

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

        self.rnn_dropout = rnn_dropout

        self.teacher_forcing = True

        inp_size = hidden_size
        if bidirectional:
            inp_size *= 2

        mlp = []
        for out_size, dropout in zip(mlp_layers, mlp_dropouts):
            mlp.append(nn.Linear(inp_size, out_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout, inplace=True))
            inp_size = out_size 
        mlp.append(nn.Linear(inp_size, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, sentence_embeddings, num_sentences, targets=None):
        batch_size = sentence_embeddings.size(0)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, 
            num_sentences.data.tolist(), 
            batch_first=True)

        packed_rnn_output, _ = self.rnn(packed_sentence_embeddings)
        mlp_input, _ = nn.utils.rnn.pad_packed_sequence(
            packed_rnn_output, 
            batch_first=False)
        mlp_input = F.dropout(
            mlp_input, p=self.rnn_dropout, training=self.training,
            inplace=True)
        logits = self.mlp(mlp_input).permute(1, 0, 2).squeeze(-1)
        return logits

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" RNNSentenceExtractor initialization started.")
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
            logger.info(" RNNSentenceExtractor initialization finished.")
