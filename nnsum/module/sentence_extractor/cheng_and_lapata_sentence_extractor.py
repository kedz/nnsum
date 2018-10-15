import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class ChengAndLapataSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, cell="gru",
                 rnn_dropout=0.0, mlp_layers=[100], mlp_dropouts=[.25]):

        super(ChengAndLapataSentenceExtractor, self).__init__()
        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))

        if cell == "gru":
            self.encoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout if num_layers > 1 else 0., 
                bidirectional=False)
            self.decoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
        elif cell == "lstm":
            self.encoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
            self.decoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
        else:
            self.encoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
            self.decoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)

        self.decoder_start = nn.Parameter(
            torch.FloatTensor(input_size).normal_())

        self.rnn_dropout = rnn_dropout

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

    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument(
            "--hidden-size", default=300, type=int)
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

    def _apply_rnn(self, rnn, packed_input, rnn_state=None, batch_first=True):
        packed_output, updated_rnn_state = rnn(packed_input, rnn_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=batch_first)
        output = F.dropout(output, p=self.rnn_dropout, training=self.training)
        return output, updated_rnn_state

    def _teacher_forcing_forward(self, sentence_embeddings, num_sentences, 
                                 targets):

        length_list = num_sentences.data.tolist()

        batch_size = sentence_embeddings.size(0)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, length_list, batch_first=True)

        encoder_output, rnn_state = self._apply_rnn(
            self.encoder_rnn, 
            packed_sentence_embeddings)
        
        weighted_decoder_input = sentence_embeddings[:,:-1] \
            * targets.view(batch_size, -1,1)[:,:-1]

        start_emb = self.decoder_start.view(1, 1, -1).repeat(batch_size, 1, 1)
        decoder_input = torch.cat([start_emb, weighted_decoder_input], 1)

        packed_decoder_input = nn.utils.rnn.pack_padded_sequence(
                decoder_input, length_list, batch_first=True)
        decoder_output, _ = self._apply_rnn(
            self.decoder_rnn, packed_decoder_input, rnn_state=rnn_state)
        mlp_input = torch.cat([encoder_output, decoder_output], 2)
        logits = self.mlp(mlp_input).squeeze(-1)
        return logits

    def _predict_forward(self, sentence_embeddings, num_sentences):
 
        length_list = num_sentences.data.tolist()

        sequence_size = sentence_embeddings.size(1)
        batch_size = sentence_embeddings.size(0)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings.permute(1, 0, 2), 
            length_list, batch_first=False)

        encoder_output, rnn_state = self._apply_rnn(
            self.encoder_rnn, 
            packed_sentence_embeddings, 
            batch_first=False)
 
        encoder_outputs = encoder_output.split(1, dim=0)
        start_emb = self.decoder_start.view(1, 1, -1).repeat(1, batch_size, 1)
        decoder_inputs = sentence_embeddings.permute(1,0,2).split(1, dim=0)

        logits = []
        decoder_input_t = start_emb
        for t in range(sequence_size):
            decoder_output_t, rnn_state = self.decoder_rnn(
                decoder_input_t, rnn_state)
            decoder_output_t = F.dropout(
                decoder_output_t, p=self.rnn_dropout, training=self.training)
            mlp_input_t = torch.cat([encoder_outputs[t], decoder_output_t], 2)
            logits_t = self.mlp(mlp_input_t)
            logits.append(logits_t)

            if t + 1 != sequence_size:
                probs_t = torch.sigmoid(logits_t)
                decoder_input_t = decoder_inputs[t] * probs_t

        logits = torch.cat(logits, 0).transpose(1, 0).squeeze(-1)
        return logits

    def forward(self, sentence_embeddings, num_sentences, targets=None):
        if self.training and self.teacher_forcing:
            return self._teacher_forcing_forward(
                sentence_embeddings, num_sentences, targets)
        else:
            return self._predict_forward(sentence_embeddings, num_sentences)

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(
                " ChengAndLapataSentenceExtractor initialization started.")
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
            logger.info(
                " ChengAndLapataSentenceExtractor initialization finished.")
