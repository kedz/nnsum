import torch
import torch.nn as nn
import torch.nn.functional as F

import nnsum.attention
from .rnn_state import RNNState
from .search_state import SearchState


class RNNDecoder(nn.Module):
    def __init__(self, embedding_context, hidden_dim=512, num_layers=1,
                 rnn_cell="GRU", dropout=0., attention="none", attn_temp=1.):
        super(RNNDecoder, self).__init__()

        rnn_cell = rnn_cell.upper()
        self._dropout = dropout
        assert rnn_cell in ["LSTM", "GRU", "RNN"]
        assert hidden_dim > 0
        assert num_layers > 0
        assert attention in ["none", "dot"]

        self._emb_ctx = embedding_context        
        self._rnn = getattr(nn, rnn_cell)(
            embedding_context.output_size, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.)

        pred_dim = hidden_dim if attention == "none" else 2 * hidden_dim
        self._predictor = nn.Sequential(
            nn.Linear(pred_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(self._emb_ctx.vocab)))
      
        if attention == "none":
            self._context_attention = None
        else:
            self._context_attention = nnsum.attention.DotAttention(
                hidden_dim, hidden_dim, temp=attn_temp)

    def forward(self, prev_rnn_state, inputs, context):

        rnn_input = self.embedding_context(inputs)
        rnn_output, rnn_state = self.rnn(rnn_input, prev_rnn_state)
        rnn_output = F.dropout(rnn_output, p=self._dropout, 
                               training=self.training)

        context_attention, weighted_context = self._compute_attention(
            context.get("encoder_output", None), 
            context.get("source_mask", None),
            rnn_output)
        
        target_logits = self._compute_logits(rnn_output, weighted_context)

        return SearchState(rnn_input=rnn_input, rnn_output=rnn_output,
                           rnn_state=RNNState.new_state(rnn_state),
                           context_attention=context_attention,
                           weighted_context=weighted_context,
                           target_logits=target_logits)

    def next_state(self, prev_state, context, compute_log_probability=False,
                   compute_output=False, compute_top_k=-1):

        next_state = self.forward(
            prev_state["rnn_state"], 
            prev_state["output"].t(),
            context)

        if compute_log_probability:
            next_state["log_probability"] = torch.log_softmax(
                next_state["target_logits"], dim=2)

        if compute_output:
            if compute_top_k > 0:
                #do top k
                if compute_log_probability:
                    #get_log_prob
                    pass
            else:
                if compute_log_probability:
                    output_lp, output = next_state["log_probability"].max(2)
                    next_state["output"] = output
                    next_state["output_log_probability"] = output_lp
                else:
                    _, output = next_state["target_logits"].max(2)
                    next_state["output"] = output

        return next_state

    def _compute_attention(self, context, context_mask, query):
        if self._context_attention:
            return self._context_attention(context, query, 
                                           context_mask=context_mask)
        else:
            return None, None

    def _compute_logits(self, rnn_output, attended_context):
        if attended_context is not None:
            predictor_input = torch.cat([rnn_output, attended_context], 2)
        else:
            predictor_input = rnn_output
        return self._predictor(predictor_input)

    @property
    def rnn(self):
        return self._rnn

    @property
    def embedding_context(self):
        return self._emb_ctx

    @property
    def predictor(self):
        return self._predictor

    def start_inputs(self, batch_size, device=None):
        return torch.tensor(
            [[self.embedding_context.vocab.start_index]] * batch_size,
            device=device)
        inputs = {n: torch.LongTensor([[v.start_index]] * batch_size)   
                  for n, v in self.embedding_context.named_vocabs.items()}
        return inputs 

    def initialize_parameters(self):
        #print(" Initializing decoder embedding context parameters.")
        self.embedding_context.initialize_parameters()
        #print(" Initializing decoder parameters.")
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 1.)
        for name, param in self._predictor.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 1.)

