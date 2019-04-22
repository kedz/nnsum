import torch


class GreedySearch(object):
    def __init__(self, decoder, init_state, init_context, max_steps=999999,
                 return_incomplete=True, compute_log_probability=False,
                 compute_cumulative_log_probability=False):

        self._decoder = decoder
        self._max_steps = max_steps
        self._return_incomplete = return_incomplete 
        self._compute_log_prob = compute_log_probability
        self._compute_cumulative_log_prob = compute_cumulative_log_probability
        
        self._init_search_state = decoder.initialize_search_state(init_state)
        self._context = decoder.initialize_context(init_context)

        self._steps = 0
        self._is_finished = False
        self._states = []
        self._outputs = []

    def next_state(self, prev_state, active_items):

        # Get next state from the decoder.
        next_state = self.decoder.next_state(
            prev_state, self.context, compute_output=True,
            compute_log_probability=self.compute_log_probability)
        
        # Mask outputs if we have already completed that batch item. 
        next_state["output"].data.view(-1).masked_fill_(
            ~active_items, self.pad_index)

        return next_state

    def check_termination(self, next_state, active_items):

        # Check for stop tokens and batch item inactive if so.
        nonstop_tokens = next_state["output"].view(-1).ne(self.stop_index)
        active_items = active_items.data.mul_(nonstop_tokens)

        return active_items

    def _collect_search_states(self, active_items):
        search_state = self._states[0]
        for next_state in self._states[1:]:
            search_state.append(next_state)
        self._states = search_state
        self._outputs = torch.cat(self._outputs, dim=1)
    
    def search(self):
        search_state = self.init_search_state
        batch_size = search_state["decoder_state"].size(1)
        active_items = search_state["decoder_state"].new(batch_size).byte() \
            .fill_(1)
        
        step_masks = []
        # Perform search until we either trigger a termination condition for
        # each batch item or we reach the maximum number of search steps.
        while self.steps < self.max_steps and not self.is_finished:
            
            inactive_items = ~active_items

            # Mask any inputs that are finished, so that greedy would 
            # be identitcal to forward passes. 
            search_state["output"].data.view(-1).masked_fill_(
                inactive_items, self.pad_index)

            step_masks.append(inactive_items)
            self._steps += 1
            search_state = self.next_state(search_state, active_items)        
            
            self._states.append(search_state)
            self._outputs.append(search_state["output"].clone())

            active_items = self.check_termination(search_state, active_items)
            self._is_finished = torch.all(~active_items)

        # Finish the search by collecting final sequences, and other 
        # stats. 
        self._collect_search_states(active_items)
        self._incomplete_items = active_items
        self._is_finished = True

        self._mask_T = torch.stack(step_masks)
        self._mask = self._mask_T.t().contiguous()

        if self.compute_cumulative_log_probability:
            seq_log_prob = self._states["log_probability"].gather(
                2, self._outputs.t().unsqueeze(2)).squeeze(2).t()
            cum_log_prob = seq_log_prob.cumsum(dim=1)
            self._states["cumulative_log_probability"] = (
                cum_log_prob, ("batch", "sequence")
            )

    @property
    def steps(self):
        return self._steps

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def compute_log_probability(self):
        return self._compute_log_prob or self._compute_cumulative_log_prob

    @property
    def compute_cumulative_log_probability(self):
        return self._compute_cumulative_log_prob

    @property
    def decoder(self):
        return self._decoder

    @property
    def init_search_state(self):
        return self._init_search_state

    @property
    def context(self):
        return self._context

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def return_incomplete(self):
        return self._return_incomplete

    @property
    def stop_index(self):
        return self.decoder.output_embedding_context.vocab.stop_index

    @property
    def pad_index(self):
        return self.decoder.output_embedding_context.vocab.pad_index

    def _mask_result(self, field, states, mask_value=0):
        result = states[field]
        if result is None:
            return None
        dim_names = states._dim_names[field]
        batch_index = dim_names.index("batch")
        seq_index = dim_names.index("sequence")
        mask_dims = [1] * len(dim_names)
        mask_dims[batch_index] = result.size(batch_index)
        mask_dims[seq_index] = result.size(seq_index)

        if batch_index < seq_index:
            mask = self._mask.view(*mask_dims)
            return result.masked_fill(mask, mask_value)
        else:
            mask = self._mask_T.view(*mask_dims)
            return result.masked_fill(mask, mask_value)

    def get_result(self, field, mask=False, mask_value=0):
        if not self.is_finished:
            self.search()

        if field == "output":
            # Output is always already masked. 
            result = self._outputs
        elif mask:
            result = self._mask_result(field, self._states,
                                       mask_value=mask_value)
        else:
            result = self._states[field]

        if not self.return_incomplete:
            from warnings import warn
            warn("Returning partial solutions is experimental.")

            batch_size = self._incomplete_items.size(0)
            dim_names = self._states._dim_names[field]
            new_view = [1 for _ in dim_names]
            new_view[dim_names.index("batch")] = batch_size
            result.data.masked_fill_(
                self._incomplete_items.view(*new_view), 0)
            
        return result
