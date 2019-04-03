import torch
from .search_algo import DecoderSearch


class GreedySearch(DecoderSearch):
    def __init__(self, decoder, init_state, context, max_steps=1000, 
                 return_incomplete=True):

        super(GreedySearch, self).__init__(decoder, init_state, context, 
                                           max_steps=max_steps,
                                           return_incomplete=return_incomplete)
       
    def next_state(self, prev_state, active_items):
        
        # Get next state from the decoder.
        next_state = self.decoder.next_state(prev_state, self.context, 
                                             compute_log_probability=True,
                                             compute_output=True)

        # Mask outputs if we have already completed that batch item.
        next_state["output"].data.view(-1).masked_fill_(
            ~active_items, self.pad_index)

        return next_state

    def check_termination(self, next_state, active_items):

        # Check for stop tokens and batch item inactive if so.
        nonstop_tokens = next_state["output"].view(-1).ne(self.stop_index)
        active_items = active_items.mul_(nonstop_tokens)

        return active_items
