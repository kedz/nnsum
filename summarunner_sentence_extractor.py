import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SummaRunnerSentenceExtractor(nn.Module):

    def __init__(self, input_size, sent_size=100, doc_size=200, 
                 num_positions=50, num_segments=4, position_size=50, 
                 segment_size=50):
        super(SummaRunnerSentenceExtractor, self).__init__()
        self.sent_rep = nn.Sequential(
            nn.Linear(input_size, sent_size), nn.ReLU())
        self.content_logits = nn.Linear(sent_size, 1)
        self.doc_rep = nn.Sequential(
            nn.Linear(sent_size, doc_size), nn.Tanh(), 
            nn.Linear(doc_size, sent_size))
        self.similarity = nn.Bilinear(sent_size, sent_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

        self.num_positions = num_positions
        self.num_segments = num_segments
        self.position_encoder = nn.Sequential(
            nn.Embedding(num_positions + 1, position_size, padding_idx=0),
            nn.Linear(position_size, 1, bias=False))
        self.segment_encoder = nn.Sequential(
            nn.Embedding(num_segments + 1, segment_size, padding_idx=0),
            nn.Linear(segment_size, 1, bias=False))

    def novelty_(self, sentence_state, summary_rep):
        sim = self.similarity(
            sentence_state.squeeze(1), F.tanh(summary_rep).squeeze(1))
        novelty = -sim.squeeze(1)
        return novelty

    def position_logits(self, length):
        batch_size = length.size(0)
        abs_pos = Variable(
            length.data.new(
                range(1, length.data[0] + 1)).view(
                    1, -1).repeat(batch_size, 1))

        chunk_size = (length.float() / self.num_segments).round().view(-1, 1)
        rel_pos = (abs_pos.float() / chunk_size).ceil().clamp(
            0, self.num_segments).long()
        abs_pos.data.clamp_(0, self.num_positions)
        pos_logits = self.position_encoder(abs_pos).squeeze(2)
        seg_logits = self.segment_encoder(rel_pos).squeeze(2)
        return pos_logits, seg_logits

    def forward(self, sentence_emb, length, encoder_output, encoder_state,
                targets=None):
        batch_size = sentence_emb.size(0)
        doc_size = sentence_emb.size(1)
        sentence_states = self.sent_rep(encoder_output)
        content_logits = self.content_logits(sentence_states).squeeze(2)

        avg_sentence = sentence_states.sum(1).div_(length.view(-1, 1).float())
        doc_rep = self.doc_rep(avg_sentence).unsqueeze(2)
        salience_logits = sentence_states.bmm(doc_rep).squeeze(2)

        sentence_states = sentence_states.split(1, dim=1)

        pos_logits, seg_logits = self.position_logits(length)

        logits = []
        summary_rep = Variable(
            sentence_states[0].data.new(sentence_states[0].size()).fill_(0))

        for step in range(doc_size):
            novelty_logits = self.novelty_(sentence_states[step], summary_rep)
            logits_step = content_logits[:, step] + salience_logits[:, step] \
                + novelty_logits + pos_logits[:, step] + seg_logits[:, step] \
                + self.bias
            
            prob = F.sigmoid(logits_step)
            
            summary_rep += sentence_states[step] * prob.view(-1, 1, 1)
            logits.append(logits_step.view(-1, 1))
        logits = torch.cat(logits, 1)
        return logits
