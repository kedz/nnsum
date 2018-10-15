import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import (NoAttention, BiLinearSoftmaxAttention, 
                         BiLinearSigmoidAttention)

import argparse


class Seq2SeqSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, 
                 cell="gru", rnn_dropout=0.0, bidirectional=False,
                 mlp_layers=[100], mlp_dropouts=[.25],
                 attention="bilinear-softmax"):

        super(Seq2SeqSentenceExtractor, self).__init__()

        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))
        if cell == "gru":
            self.encoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
            self.decoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
        elif cell == "lstm":
            self.encoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
            self.decoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
        else:
            self.encoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
            self.decoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)

        self.decoder_start = nn.Parameter(
            torch.FloatTensor(input_size).normal_())

        self.rnn_dropout = rnn_dropout

        if attention == "bilinear-softmax":
            self.attention = BiLinearSoftmaxAttention()
        elif attention == "bilinear-sigmoid":
            self.attention = BiLinearSigmoidAttention()
        elif attention == "none":
            self.attention = NoAttention()
        else:
            raise Exception("attention must be 'none', 'bilinear-softmax', "
                            "or 'bilinear-sigmoid'.")

        self.teacher_forcing = True

        inp_size = hidden_size
        if bidirectional:
            inp_size *= 2

        if attention != "none":
            inp_size *= 2
        mlp = []
        for out_size, dropout in zip(mlp_layers, mlp_dropouts):
            mlp.append(nn.Linear(inp_size, out_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout, inplace=True))
            inp_size = out_size 
        mlp.append(nn.Linear(inp_size, 1))
        self.mlp = nn.Sequential(*mlp)

    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument(
            "--hidden-size", default=300, type=int)
        parser.add_argument(
            "--bidirectional", action="store_true", default=False)
        parser.add_argument(
            "--rnn-dropout", default=.25, type=float)
        parser.add_argument(
            "--num-layers", default=1, type=int)
        parser.add_argument("--cell", choices=["rnn", "gru", "lstm"],
                            default="gru", type=str)
        parser.add_argument(
            "--mlp-layers", default=[100], type=int, nargs="+")
        parser.add_argument(
            "--mlp-dropouts", default=[.25], type=float, nargs="+")
        return parser

    def _start_decoder(self, batch_size, rnn_state):
        start_emb = self.decoder_start.view(1, 1, -1).repeat(1, batch_size, 1)
        _, updated_rnn_state = self.decoder_rnn(start_emb, rnn_state)
        return updated_rnn_state

    def _apply_rnn(self, rnn, packed_input, rnn_state=None):
        packed_output, updated_rnn_state = rnn(packed_input, rnn_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True)
        output = F.dropout(
            output, p=self.rnn_dropout, training=self.training, inplace=True)
        return output, updated_rnn_state

    def forward(self, sentence_embeddings, num_sentences, targets=None):

        batch_size = sentence_embeddings.size(0)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, 
            num_sentences.data.tolist(), 
            batch_first=True)

        encoder_output, rnn_state = self._apply_rnn(
            self.encoder_rnn, 
            packed_sentence_embeddings)

        rnn_state = self._start_decoder(batch_size, rnn_state)
        
        decoder_output, rnn_state = self._apply_rnn(
            self.decoder_rnn,
            packed_sentence_embeddings,
            rnn_state=rnn_state)

        mlp_input, scores = self.attention(
            encoder_output, decoder_output, num_sentences)

        return self.mlp(mlp_input).squeeze(2), scores


    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" Seq2SeqSentenceExtractor initialization started.")
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
            logger.info(" Seq2SeqSentenceExtractor initialization finished.")
