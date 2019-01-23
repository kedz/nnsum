import torch
from .search_state import SearchState


class DecoderSearch(object):
    def __init__(self, decoder, encoder_state, context, max_steps=10000,
                 context_mask=None, return_incomplete=True):

        self._decoder = decoder
        self._batch_size = encoder_state.size(1)
        self._context, self._context_mask = self._initialize_context(
            context, context_mask)
        self._is_finished = False
        self._steps = 0
        self._max_steps = max_steps
        self._return_incomplete = return_incomplete
        
        self._init_search_state = self._initialize_search_state(encoder_state)
        self._state_history = []
        self._results_cache = {}

    @property
    def decoder(self):
        return self._decoder

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def context(self):
        return self._context

    @property
    def context_mask(self):
        return self._context_mask

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def steps(self):
        return self._steps

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def return_incomplete(self):
        return self._return_incomplete

    @property
    def pad_index(self):
        return self.decoder.embedding_context.vocab.pad_index

    @property
    def start_index(self):
        return self.decoder.embedding_context.vocab.start_index
        
    @property
    def stop_index(self):
        return self.decoder.embedding_context.vocab.stop_index
    
    @property
    def unknown_index(self):
        return self.decoder.embedding_context.vocab.unknown_index

    def _initialize_context(self, context, context_mask):
        return context, context_mask

    def _initialize_search_state(self, encoder_state):
        outputs = self.decoder.start_inputs(self.batch_size).t()
        return SearchState(rnn_state=encoder_state, outputs=outputs)

    def search(self):
       
        search_state = self._init_search_state
        active_items = torch.ByteTensor(
            self.batch_size, device=search_state["rnn_state"].device).fill_(1)

        # Perform search until we either trigger a termination condition for
        # each batch item or we reach the maximum number of search steps.
        while self.steps < self.max_steps and not self.is_finished:
            
            self._steps += 1
            search_state = self.next_state(search_state, active_items)        
            self._state_history.append(search_state)

            active_items = self.check_termination(search_state, active_items)
            self._is_finished = torch.all(~active_items)

        # Finish the search by collecting final sequences, and other 
        # stats. 
        self._collect_search_states(active_items)
        self._is_finished = True

    def next_state(self, prev_state):
        raise Exception("Not Implemented!")

    def terminate_search(self, next_state):
        raise Exception("Not Implemented!")

    def _collect_search_states(self, active_items):
        search_state = self._state_history[0]
        for next_state in self._state_history[1:]:
            search_state.append(next_state)
        self._state_history = search_state
        self._state_history["active_items"] = active_items

    def get_results(self, *fields):
        return tuple([self.get_result(field) for field in fields])

    def get_result(self, field):
        if not self.is_finished:
            self.search()

        result = self._results_cache.get(field, None)
        if result is None:
            result = getattr(self, "_collect_{}".format(field))()
            self._results_cache[field] = result
        return result
 
    def _collect_outputs(self):
        outputs = self._state_history["outputs"]

        # (Optionally) Mask incomplete sequences with pad index. 
        # Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1)
            outputs.data.masked_fill_(mask, self.pad_index)
        return outputs

    def _collect_logits(self):

        logits = self._state_history["logits"]

        # (Optionally) Mask incomplete sequences with 0. Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1, 1)
            logits.data.masked_fill_(mask, 0.)

        # Set logits to uniformly 0 for pad tokens.
        mask = self.get_result("outputs").eq(self.pad_index).unsqueeze(-1)
        logits.data.masked_fill_(mask, 0.)

        return logits

    def _collect_log_probability(self):
        result = self._state_history["log_probability"]

        # (Optionally) Mask incomplete sequences with 0. Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1, 1)
            result.data.masked_fill_(mask, 0.)

        # Set logits to uniformly 0 for pad tokens.
        mask = self.get_result("outputs").eq(self.pad_index).unsqueeze(-1)
        result.data.masked_fill_(mask, 0.)

        return result

    def _collect_context_attention(self):

        ctx_attn = self._state_history["context_attention"]

        # (Optionally) Mask incomplete sequences with 0. Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1, 1)
            ctx_attn.data.masked_fill_(mask, 0.)

        # Set attention to uniformly 0 for pad tokens.
        mask = self.get_result("outputs").eq(self.pad_index).unsqueeze(-1)
        ctx_attn.data.masked_fill_(mask, 0.)
        return ctx_attn

    def _collect_rnn_outputs(self):

        result = self._state_history["rnn_outputs"]

        # (Optionally) Mask incomplete sequences with 0. Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1, 1)
            result.data.masked_fill_(mask, 0.)

        # Set attention to uniformly 0 for pad tokens.
        mask = self.get_result("outputs").eq(self.pad_index).unsqueeze(-1)
        result.data.masked_fill_(mask, 0.)
        return result

    def _collect_output_log_probability(self):
        outputs = self.get_result("outputs").unsqueeze(-1)
        mask = outputs.eq(self.pad_index)
        log_probs = self._state_history["log_probability"]
        result = log_probs.gather(2, outputs)
        result.data.masked_fill_(mask, 0.)
        result = result.squeeze(-1)

        # (Optionally) Mask incomplete sequences with 0. Skip this by default.
        if not self.return_incomplete:
            mask = self._state_history["active_items"].view(1, -1)
            result.data.masked_fill_(mask, 0.)

        return result
