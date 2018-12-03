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

    def decode(self, context, state, max_steps=1000, return_log_probs=False):
        batch_size = context.size(0)
        
        start_idx = self.embedding_context.vocab.start_index
        stop_idx = self.embedding_context.vocab.stop_index
        pad_idx = self.embedding_context.vocab.pad_index

        inputs = context.data.new(
            batch_size).long().fill_(start_idx).view(-1, 1)

        predicted_tokens = []
        token_log_probs = []
        
        active_items = inputs.ne(stop_idx).view(-1)
        for step in range(max_steps):
            logits, attn, state = self.forward(inputs, context, state)
            a, next_tokens = logits.max(2)
            
            if return_log_probs:
                
                lp_step = logits.gather(2, next_tokens.view(1, -1, 1)) \
                    - torch.logsumexp(logits, dim=2, keepdim=True)
                lp_step.data.view(-1).masked_fill_(~active_items, 0)
                token_log_probs.append(lp_step)  
            
            inputs = next_tokens.t()
            inputs.view(-1).data.masked_fill_(~active_items, pad_idx)
            predicted_tokens.append(inputs)
            active_items = active_items * inputs.view(-1).ne(stop_idx)
            if torch.all(~active_items):
                break
        predicted_tokens = torch.cat(predicted_tokens, dim=1)

        if return_log_probs:
            return predicted_tokens, torch.cat(token_log_probs, 0)
        else:
            return predicted_tokens

    def start_inputs(self, batch_size):
        inputs = {n: torch.LongTensor([[v.start_index]] * batch_size)   
                  for n, v in self.embedding_context.named_vocabs.items()}
        return inputs 

