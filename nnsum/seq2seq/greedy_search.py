import torch
from .search_algo import DecoderSearch


class GreedySearch(DecoderSearch):
    def __init__(self, decoder, init_state, context, max_steps=1000, 
                 context_mask=None, return_incomplete=True):

        super(GreedySearch, self).__init__(decoder, init_state, context, 
                                           max_steps=max_steps,
                                           context_mask=context_mask,
                                           return_incomplete=return_incomplete)
       
    def next_state(self, prev_state, active_items):
        next_state = self.decoder.next_state(
            prev_state["rnn_state"], 
            inputs=prev_state["outputs"].t(), 
            context=self.context, context_mask=self.context_mask,
            compute_log_probability=True)
        next_tokens = next_state["logits"].max(2)[1]
        next_tokens.data.view(-1).masked_fill_(~active_items, self.pad_index)
        next_state["outputs"] = next_tokens
        return next_state

    def check_termination(self, next_state, active_items):
        nonstop_tokens = next_state["outputs"].view(-1).ne(self.stop_index)
        active_items = active_items.mul_(nonstop_tokens)
        return active_items
