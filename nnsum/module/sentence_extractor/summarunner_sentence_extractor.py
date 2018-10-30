import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class SummaRunnerSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, 
                 bidirectional=True, cell="gru", rnn_dropout=0.0,
                 sentence_size=100, document_size=200,
                 segments=4, max_position_weights=25,
                 segment_size=50, position_size=50):

        super(SummaRunnerSentenceExtractor, self).__init__()

        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))
        if cell == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.) 
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
        else:
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)

        self.rnn_dropout = rnn_dropout

        self.teacher_forcing = True

        inp_size = hidden_size
        if bidirectional:
            inp_size *= 2

        self.sentence_rep = nn.Sequential(
            nn.Linear(inp_size, sentence_size), nn.ReLU())
        self.content_logits = nn.Linear(sentence_size, 1)
        self.document_rep = nn.Sequential(
            nn.Linear(sentence_size, document_size), 
            nn.Tanh(), 
            nn.Linear(document_size, sentence_size))
        self.similarity = nn.Bilinear(
            sentence_size, sentence_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

        self.max_position_weights = max_position_weights
        self.segments = segments
        self.position_encoder = nn.Sequential(
            nn.Embedding(max_position_weights + 1, position_size, 
                         padding_idx=0),
            nn.Linear(position_size, 1, bias=False))
        self.segment_encoder = nn.Sequential(
            nn.Embedding(segments + 1, segment_size, padding_idx=0),
            nn.Linear(segment_size, 1, bias=False))

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
            "--sentence-size", type=int, default=100, required=False)
        parser.add_argument(
            "--document-size", type=int, default=100, required=False)
        parser.add_argument(
            "--segments", type=int, default=4, required=False)
        parser.add_argument(
            "--max-position-weights", type=int, default=50, required=False)
        parser.add_argument(
            "--segment-size", type=int, default=16, required=False)
        parser.add_argument(
            "--position-size", type=int, default=16, required=False)
        return parser

    def novelty(self, sentence_state, summary_rep):
        sim = self.similarity(
            sentence_state.squeeze(1), torch.tanh(summary_rep).squeeze(1))
        novelty = -sim.squeeze(1)
        return novelty

    def position_logits(self, length):
        batch_size = length.size(0)
        abs_pos = torch.arange(
            1, length.data[0].item() + 1, device=length.device)\
            .view(1, -1).repeat(batch_size, 1)

        chunk_size = (length.float() / self.segments).round().view(-1, 1)
        rel_pos = (abs_pos.float() / chunk_size).ceil().clamp(
            0, self.segments).long()

        abs_pos.data.clamp_(0, self.max_position_weights)
        pos_logits = self.position_encoder(abs_pos).squeeze(2)
        seg_logits = self.segment_encoder(rel_pos).squeeze(2)
        return pos_logits, seg_logits

    def forward(self, sentence_embeddings, num_sentences, targets=None):

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, 
            num_sentences.data.tolist(), 
            batch_first=True)

        rnn_output_packed, _ = self.rnn(packed_sentence_embeddings)

        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_output_packed, 
            batch_first=True)

        rnn_output = F.dropout(rnn_output, p=self.rnn_dropout, inplace=True,
                               training=self.training)

        sentence_states = self.sentence_rep(rnn_output)
        content_logits = self.content_logits(sentence_states).squeeze(2)

        avg_sentence = sentence_states.sum(1).div_(
            num_sentences.view(-1, 1).float())
        doc_rep = self.document_rep(avg_sentence).unsqueeze(2)
        salience_logits = sentence_states.bmm(doc_rep).squeeze(2)
        pos_logits, seg_logits = self.position_logits(num_sentences)

        static_logits = content_logits + salience_logits + pos_logits \
            + seg_logits + self.bias.unsqueeze(0)
        
        sentence_states = sentence_states.split(1, dim=1)
        summary_rep = torch.zeros_like(sentence_states[0])
        logits = []
        for step in range(num_sentences[0].item()):
            novelty_logits = self.novelty(sentence_states[step], summary_rep)
            logits_step = static_logits[:, step] + novelty_logits
            
            prob = torch.sigmoid(logits_step)
            
            summary_rep += sentence_states[step] * prob.view(-1, 1, 1)
            logits.append(logits_step.view(-1, 1))
        logits = torch.cat(logits, 1)
        return logits

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(
                " SummaRunnerSentenceExtractor initialization started.")
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
                " SummaRunnerSentenceExtractor initialization finished.")
