import torch

import numpy as np

from torch_scatter import scatter_add


class BeamSearch(object):
    def __init__(self, decoder, init_state, init_context, max_steps=999999,
                 beam_size=4, return_incomplete=True, sort_by_score=True, 
                 rescoring_func=None):

        self._decoder = decoder
        self._max_steps = max_steps
        self._return_incomplete = return_incomplete 
            
        self._beam_size = beam_size
        self._sort_by_score = sort_by_score
        
        self._init_search_state = self._initialize_search_state(init_state)
        self._context = self._initialize_context(init_context)
        self._batch_size = (
            self.init_search_state["decoder_state"].size(1) // self.beam_size
        )

        self._device = self.init_search_state["decoder_state"].device
        self._steps = 0
        self._is_finished = False
        self._states = []
        self._beam_scores = [list() for _ in range(self.batch_size)]
        self._num_complete = self.init_search_state["decoder_state"].new()\
            .long().new(self.batch_size).fill_(0)
        self._terminal_info = [list() for _ in range(self.batch_size)]


    def _initialize_search_state(self, init_state):
        
        init_state = self.decoder.initialize_search_state(init_state)
        beam_state = init_state.repeat_dim("batch", self.beam_size)

        batch_sz = init_state['decoder_state'].size(1)
        beam_sz = self.beam_size
        device = init_state['decoder_state'].device

        # Start the first beam of each batch with 0 log prob, and all others 
        # with -inf.
        beam_state["cumulative_log_probability"] = (
            self._initialize_sequence_log_probs(batch_sz, beam_sz, device), 
            ("sequence", "batch", "placeholder")
        )

        # At the first time step no sequences have been terminated so this mask
        # is all 0s. 
        beam_state["terminal_mask"] = (
            init_state["decoder_state"].new().byte().new(
                1, batch_sz * beam_sz, 1).fill_(0),
            ("sequence", "batch", "placeholder"),
        )   

        return beam_state

    def _initialize_sequence_log_probs(self, batch_size, beam_size, device):
        lp = torch.FloatTensor(1, batch_size, beam_size, 1)
        if "cuda" in str(device):
            lp = lp.cuda(device)
        lp.data.fill_(0)
        lp.data[:,:,1:].fill_(float("-inf"))
        return lp.view(1, batch_size * beam_size, 1)

    def _initialize_context(self, init_context):
        # This is should be more progammatic in the future.
        
        beam_context = {}
        
        context = self.decoder.initialize_context(init_context)
        if context.get("encoder_output", None) is not None:
            batch_size, seq_size, hid_size = context["encoder_output"].size()
            beam_context["encoder_output"] = context["encoder_output"] \
                .unsqueeze(1).repeat(1, self.beam_size, 1, 1) \
                .view(batch_size * self.beam_size, seq_size, hid_size)
            
        if context.get("source_mask", None) is not None:
            beam_context["source_mask"] = context["source_mask"] \
                .unsqueeze(1).repeat(1, self.beam_size, 1) \
                .view(batch_size * self.beam_size, seq_size)
            
        if context.get("source_extended_vocab_map", None) is not None:
            beam_context["source_extended_vocab_map"] = \
                context["source_extended_vocab_map"] \
                .unsqueeze(1).repeat(1, self.beam_size, 1) \
                .view(batch_size * self.beam_size, seq_size)
               
            beam_context["extended_vocab"] = context["extended_vocab"]

        if context.get("controls", None) is not None:
            beam_ctrls = {}
            for name, ctrl in context["controls"].items():
                beam_ctrls[name] = ctrl.view(-1, 1).repeat(1, self.beam_size) \
                    .view(-1)
            beam_context["controls"] = beam_ctrls

        return beam_context

    def _next_candidates(self, log_probs, candidates):
        # TODO seq_lps should really be called cumulative log probs.

        # flat_beam_lps (batch size x (beam size ** 2))
        flat_beam_lps = log_probs.view(self.batch_size, -1)

        flat_beam_scores = flat_beam_lps / (self.steps + 1)

        # beam_seq_scores (batch size x beam size)
        # relative_indices (batch_size x beam size)
        beam_seq_scores, relative_indices = torch.topk(
            flat_beam_scores, k=self.beam_size, dim=1)

        # beam_seq_lps (batch size x beam size)
        beam_seq_lps = flat_beam_lps.gather(1, relative_indices)

        # TODO make these ahead of time. 
        offset1 = (torch.arange(self.batch_size, device=beam_seq_lps.device) * self.beam_size) \
            .view(self.batch_size, 1)
        offset2 = offset1 * self.beam_size
       
        beam_indexing = ((relative_indices // self.beam_size) + offset1) \
            .view(-1)

        # beam_seq_lps (1 x (batch_size * beam_size) x 1)
        beam_seq_lps = beam_seq_lps \
            .view(1, self.batch_size * self.beam_size, 1)
        
        # beam_seq_scores (1 x (batch_size * beam_size) x 1)
        beam_seq_scores = beam_seq_scores \
            .view(1, self.batch_size * self.beam_size, 1)

        # next_output ((batch size * beam size) x 1)
        next_candidate_indices = (relative_indices + offset2).view(-1)
        next_output = candidates.view(-1)[next_candidate_indices].view(-1, 1)

        return beam_seq_lps, beam_seq_scores, next_output, beam_indexing

    def next_state(self, prev_state, active_items):

        # Get next state from the decoder.
        next_state = self.decoder.next_state(
            prev_state, self.context,
            compute_log_probability=True)

        # Compute the top beam_size next outputs for each beam item.
        # topk_lps (1 x batch size x beam size x beam size)
        # candidate_outputs (1 x batch size x beam size x beam size)
        topk_lps, candidate_outputs = torch.topk(
            next_state["log_probability"] \
                .view(1, self.batch_size, self.beam_size, -1),
            k=self.beam_size, dim=3)

        # If any sequence was completed last step, we should mask it's log
        # prob so that we don't generate from the terminal token.
        # slp (1 x batch_size x beam size x 1) 
        slp = prev_state["cumulative_log_probability"] \
            .masked_fill(prev_state["terminal_mask"], float("-inf")) \
            .view(1, self.batch_size, self.beam_size, 1)

        # Combine next step log probs with the previous sequences cumulative
        # log probs, i.e.
        #     log P(y_t) = log P(y_<t) + log P(y_t)
        # candidate_log_probs (1 x batch size x beam size x beam size)
        candidate_log_probs = slp + topk_lps

        # Rerank and select the beam_size best options from the available 
        # beam_size ** 2 candidates.
        # b_seq_lps (1 x (batch size * beam size) x 1)
        # b_scores (1 x (batch size * beam size) x 1)
        # b_output ((batch size * beam size) x 1)
        # b_indices ((batch size * beam size))
        b_seq_lps, b_scores, b_output, b_indices = self._next_candidates(
            candidate_log_probs, candidate_outputs)

        next_state.stage_indexing("batch", b_indices)

        next_state["output"] = (b_output, ("batch", "sequence"))
        next_state["cumulative_log_probability"] = (
            b_seq_lps, ("sequence", "batch", "placeholder")
        )
        next_state["beam_score"] = (
            b_scores, ("sequence", "batch", "placeholder")
        )
        next_state["beam_indices"] = (b_indices, ("batch"))

        return next_state

    def check_termination(self, next_state, active_items):
        
        next_output = next_state["output"] \
            .view(self.batch_size, self.beam_size)

        is_complete = next_output.eq(self.stop_index)
        complete_indices = np.where(is_complete.cpu().data.numpy())

        for batch, beam in zip(*complete_indices):
            if self._num_complete[batch] == self.beam_size:
                continue
            else:
                self._num_complete[batch] += 1

                # Store step and beam that finished so we can retrace it
                # later and recover arbitrary search state item.
                self._terminal_info[batch].append(
                    (self.steps, beam + batch * self.beam_size))
                
                IDX = batch * self.beam_size + beam
                self._beam_scores[batch].append(
                    next_state["beam_score"][0, IDX, 0].view(1))
        
        next_state["terminal_mask"] = (
            is_complete.view(1, self.batch_size * self.beam_size, 1),
            ("sequence", "batch", "placeholder"),
        )   
        active_items = self._num_complete < self.beam_size

        return active_items

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
            mask = self._selector_mask.view(*mask_dims)
            return result.masked_fill(mask, mask_value)
        else:
            mask = self._selector_mask_T.view(*mask_dims)
            return result.masked_fill(mask, mask_value)

    def get_result(self, field, mask=False, mask_value=0):
        if not self.is_finished:
            self.search()
        
        if mask:
            result = self._mask_result(field, self._states,
                                       mask_value=mask_value)
        else:
            result = self._states[field]

        if not self.return_incomplete:
            raise Exception("Incomplete returns not yet implemented.")
        
        if field == "output":
            return result.view(self.batch_size, self.beam_size, -1)

        return result


        if field == "output":
            result = self._outputs
        else:
            result = self._states[field]

        if not self.return_incomplete:
            batch_size = self._incomplete_items.size(0)
            dim_names = self._states._dim_names[field]
            new_view = [1 for _ in dim_names]
            new_view[dim_names.index("batch")] = batch_size
            result.data.masked_fill_(
                self._incomplete_items.view(*new_view), 0)
            
        return result

    def search(self):

        # Start search with init_search_state and all batches active.
        search_state = self.init_search_state
        active_items = search_state["decoder_state"].new().byte()\
            .new(self.batch_size).fill_(1)

        # Perform search until we either trigger a termination condition for
        # each batch item or we reach the maximum number of search steps.
        while self.steps < self.max_steps and not self.is_finished:
            
            search_state = self.next_state(search_state, active_items)        
            active_items = self.check_termination(search_state, active_items)
            self._is_finished = torch.all(~active_items)

            self._states.append(search_state)
            self._steps += 1

        # Finish the search by collecting final sequences, and other 
        # stats. 
        self._collect_search_states(active_items)
        self._incomplete_items = active_items
        self._is_finished = True

    def _collect_search_states(self, active_items):

        last_state = self._states[-1]
        last_step = self.steps - 1
        for batch in range(self.batch_size):
            beam = 0 
            while len(self._beam_scores[batch]) < self.beam_size:
                IDX = batch * self.beam_size + beam
                self._beam_scores[batch].append(
                    last_state["beam_score"][0, IDX, 0].view(1))
                self._terminal_info[batch].append(
                    (last_step, beam + batch * self.beam_size))
                beam += 1

        # TODO consider removing beam indices from state
        beam_indices = torch.stack([state["beam_indices"] 
                                    for state in self._states])

        self._beam_scores = torch.stack([torch.cat(bs)
                                         for bs in self._beam_scores])
        
        lengths = self._states[0]["output"].new(
            [[step + 1 for step, beam in self._terminal_info[batch]]
             for batch in range(self.batch_size)])
        
        selector = self._states[0]["output"].new(
            self.batch_size, self.beam_size, lengths.max())
        mask = selector.new().byte().new(selector.size()).fill_(1)

        for batch in range(self.batch_size):
            for beam in range(self.beam_size):
                step, real_beam = self._terminal_info[batch][beam]
                mask[batch, beam,:step + 1].fill_(0)
                self._collect_beam(batch, real_beam, step, 
                                   beam_indices,
                                   selector[batch, beam])
        selector = selector.view(self.batch_size * self.beam_size, -1)

        ## RESORTING HERE ##
        if self.sort_by_score:
            self._beam_scores, I = torch.sort(self._beam_scores, dim=1, descending=True)
            offset1 = (torch.arange(self.batch_size, device=I.device) * self.beam_size) \
                .view(self.batch_size, 1)
            II = I + offset1
            selector = selector[II.view(-1)]
            mask = mask.view(self.batch_size * self.beam_size,-1)[II].view(self.batch_size, self.beam_size, -1)
            lengths = lengths.gather(1, I)
        ## 
 
        for step, sel_step in enumerate(selector.split(1, dim=1)):
            self._states[step].stage_indexing("batch", sel_step.view(-1))

        states = self._states[0]
        for state in self._states[1:]:
            states.append(state)

        self._states = states
        self._selector = selector
        self._lengths = lengths
        self._selector_mask = mask.view(self.batch_size * self.beam_size, -1)
        self._selector_mask_T = self._selector_mask.t().contiguous()

        return 

    def _collect_beam(self, batch, beam, step, beam_indices,
                      selector_out):        
        selection = [0] * beam_indices.size(0)
        selector_out[step + 1:].fill_(0) 
        while step >= 0:
            selection[step] = beam
            selector_out[step].fill_(beam)
            next_beam = beam_indices[step, beam].item()
            beam = next_beam
            step -= 1

    @property
    def steps(self):
        return self._steps

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def device(self):
        return self._device

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

    @property
    def beam_scores(self):
        if self.is_finished:
            return self._beam_scores
        else:
            raise Exception("Run search() first to get beam scores.")
    
    @property
    def sort_by_score(self):
        return self._sort_by_score
