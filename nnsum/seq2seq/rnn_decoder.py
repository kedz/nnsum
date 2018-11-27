import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(self, embedding_context, 
                 hidden_dim=512, num_layers=1,
                 rnn_cell="GRU"):
        super(RNNDecoder, self).__init__()

        rnn_cell = rnn_cell.upper()
        assert rnn_cell in ["LSTM", "GRU", "RNN"]
        assert hidden_dim > 0
        assert num_layers > 0

        self._emb_ctx = embedding_context        
        self._rnn = getattr(nn, rnn_cell)(
            embedding_context.output_size, hidden_dim, num_layers=num_layers)
        self._predictor = nn.Linear(hidden_dim, len(self._emb_ctx.vocab))
       
    @property
    def rnn(self):
        return self._rnn

    @property
    def embedding_context(self):
        return self._emb_ctx

    @property
    def predictor(self):
        return self._predictor

    def forward(self, inputs, context, state):
        decoder_output, state = self._rnn(self._emb_ctx(inputs), state)
        logits = self._predictor(decoder_output)
        return logits, {}, state

    def predict(self, context, state):
        batch_size = context.size(0)

        max_steps = 50
        
        start_idx = self.embedding_context.vocab.start_index
        stop_idx = self.embedding_context.vocab.stop_index
        pad_idx = self.embedding_context.vocab.pad_index

        inputs = context.data.new(
            batch_size).long().fill_(start_idx).view(-1, 1)

        predicted_tokens = []
        
        active_items = inputs.ne(stop_idx).view(-1)
        for step in range(max_steps):
            logits, attn, state = self.forward(inputs, context, state)
            a, next_tokens = logits.max(2)
            inputs = next_tokens.t()
            active_items = active_items * inputs.view(-1).ne(stop_idx)
            inputs.view(-1).data.masked_fill_(~active_items, pad_idx)
            if torch.all(~active_items):
                break
            predicted_tokens.append(inputs)
        predicted_tokens = torch.cat(predicted_tokens, dim=1)
        return predicted_tokens
