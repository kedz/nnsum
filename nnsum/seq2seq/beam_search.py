import torch


class BeamSearch(object):
    def __init__(self, decoder, init_state, context, beam_size=8,
                 max_steps=1000):

        self._batch_size = batch_size = context.size(0)
        total_beam_size = batch_size * beam_size

        self._decoder = decoder
        self._finished = False
        self._max_steps = max_steps
        self._current_step = 0
        self._beam_size = beam_size
        self._stop_index = decoder.embedding_context.vocab.stop_index
        self._pad_index = decoder.embedding_context.vocab.pad_index

        self._index_offset = torch.arange(
            0, self._batch_size, device=context.device).view(-1, 1)
        self._index_offset.mul_(self._beam_size)
        
        self._context = self._initialize_context(context)
        self._state = self._initialize_state(init_state)
        
        self._completed_sequences = [list() for x in range(batch_size)]
        self._completed_lengths = torch.LongTensor(
            batch_size, beam_size).fill_(0)
        self._completed_log_probs = [list() for x in range(batch_size)]
        self._completed_scores = [list() for x in range(batch_size)]

        self._log_probs = context.data.new(batch_size, beam_size, 1).fill_(0)
        self._inputs = self._decoder.start_inputs(total_beam_size)            
        self._history = None
        self._batch_complete = torch.ByteTensor(batch_size).fill_(0)
        
    @property
    def finished(self):
        return self._finished

    def _initialize_state(self, init_state):
        layers, batch_size, state_size = init_state.size()
        beam_size = self._beam_size
        return init_state.unsqueeze(2).repeat(1, 1, beam_size, 1).view(
            layers, batch_size * beam_size, state_size)

    def _initialize_context(self, context):
        batch_size, ctx_len, ctx_size = context.size()
        beam_size = self._beam_size
        return context.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            batch_size * beam_size, ctx_len, ctx_size)

    def _rescore_beam(self, log_probs, history, next_tokens):
        return log_probs / self._current_step

    def _select_candidates(self, scores, candidates, log_probs, state):

        next_scores, top_idxs = torch.topk(
            scores.view(self._batch_size, -1), 
            k=self._beam_size,
            dim=1)

        candidates = candidates.view(self._batch_size, self._beam_size ** 2)
        log_probs = log_probs.view(self._batch_size, self._beam_size ** 2)
        next_tokens = candidates.gather(1, top_idxs).view(-1, 1)
        next_log_probs = log_probs.gather(1, top_idxs).unsqueeze(-1)
        next_beam_indxs = top_idxs / self._beam_size + self._index_offset

        next_state = state[:,next_beam_indxs.view(-1)]

        if self._current_step == 1:
            self._history = next_tokens
        else:

            self._history = torch.cat(
                [self._history[next_beam_indxs.view(-1)], next_tokens], 1)

        return next_tokens, next_log_probs, next_state, next_scores.view(-1)

    def next_step(self):
        self._current_step += 1

        logits, attn, next_state = self._decoder(
            self._inputs, self._context, self._state)
        log_probs = torch.log_softmax(logits.squeeze(0), dim=1).view(
            self._batch_size, self._beam_size, -1)
        
        topk_lps, topk_idxs = torch.topk(log_probs, k=self._beam_size, dim=2)
        
        next_log_probs = self._log_probs + topk_lps
        scores = self._rescore_beam(next_log_probs, self._history, topk_idxs)

        if self._current_step == 1:
            scores[:,1:].data.fill_(float("-inf"))

        (next_inputs, next_log_probs, 
         next_state, next_scores) = self._select_candidates(
            scores, topk_idxs, next_log_probs, next_state)

        is_complete = next_inputs.view(-1).eq(self._stop_index)

        if torch.any(is_complete):
            for i, i_is_complete in enumerate(is_complete):
                if not i_is_complete:
                    continue
                
                batch = i // self._beam_size
                if len(self._completed_sequences[batch]) == self._beam_size:
                    continue

                num_complete = len(self._completed_sequences[batch])
                self._completed_sequences[batch].append(self._history[i])
                self._completed_log_probs[batch].append(
                    next_log_probs.view(-1)[i].clone())
                next_log_probs.view(-1)[i].data.fill_(float("-inf"))
                self._completed_lengths[batch, num_complete] = self._current_step
                self._completed_scores[batch].append(next_scores[i])
                if len(self._completed_sequences[batch]) == self._beam_size:
                    self._batch_complete[batch] = 1

        self._finished = torch.all(self._batch_complete)

        self._inputs = next_inputs
        self._log_probs = next_log_probs
        self._state = next_state
        self._scores = next_scores

    def _add_incomplete_to_beam(self):
        history = self._history
        lps = self._log_probs.view(-1)
        cur_step = self._current_step
        scores = self._scores

        for batch in range(self._batch_size):
            beam = 0
            num_compl = len(self._completed_sequences[batch])
            while num_compl < self._beam_size:
                idx = batch * self._beam_size + beam 
                if self._history[idx,-1].ne(self._stop_index):
                    self._completed_sequences[batch].append(history[idx])
                    self._completed_log_probs[batch].append(lps[idx])
                    self._completed_lengths[batch, num_compl] = cur_step
                    self._completed_scores[batch].append(scores[idx])
                    num_compl += 1
                beam += 1

    def _collect_beam(self):
        max_len = self._completed_lengths.max().item()
        if max_len == 0:
            max_len = 1
        beam_sz = self._beam_size
        batch_sz = self._batch_size
        lengths = self._completed_lengths
        sequences = self._completed_lengths.new(
            batch_sz, beam_sz, max_len).fill_(self._pad_index)
        lps = self._completed_log_probs
        scores = self._completed_scores

        for i, batch in enumerate(self._completed_sequences):
            for j, candidate in enumerate(batch):
                sequences[i,j,:lengths[i,j]].copy_(candidate)
            if len(lps[i]) < beam_sz:
                remainder = beam_sz - len(lps[i])
                lps[i].extend([torch.tensor(float("-inf"))] * remainder)
                scores[i].extend([torch.tensor(float("-inf"))] * remainder)
            lps[i] = torch.stack(lps[i])
            scores[i] = torch.stack(scores[i])
        self._completed_log_probs = torch.stack(lps)
        self._completed_scores = torch.stack(scores)
        self._completed_sequences = sequences

    def search(self, return_incomplete=False):
       
        # Perform beam search until either we find beam_size completed 
        # sequences for each batch item or we reach the maximum number of 
        # search steps.
        while self._current_step < self._max_steps and not self.finished:
            self.next_step()        

        # If the beam search reached the maximum number of steps but the
        # number of completed results is not equal to the beam size, add the 
        # current best scoring sequences to fill out the beam.
        if not self.finished and return_incomplete:
            self._add_incomplete_to_beam()

        # Finish the search by collecting final beam sequences, and other 
        # stats. 
        self._collect_beam()                        
        self._finished = True

    @property
    def candidates(self):
        if self.finished:
            return self._completed_sequences
        else:
            return None

    @property
    def scores(self):
        if self.finished:
            return self._completed_scores
        else:
            return None

    @property
    def log_probs(self):
        if self.finished:
            return self._completed_log_probs
        else:
            return None

    @property
    def lengths(self):
        if self.finished:
            return self._completed_lengths
        else:
            return None

    def sort_by_score(self):
        if not self.finished:
            raise Exception(
                "BeamSearch must be finished before it can be sorted.")
        self._completed_scores, indices = torch.sort(
            self._completed_scores, 1, descending=True)

        self._completed_log_probs = self._completed_log_probs.gather(
            1, indices)
        self._completed_lengths = self._completed_lengths.gather(1, indices)

        comp_seqs = self._completed_sequences.view(
            self._batch_size * self._beam_size, -1)
        st_comp_seqs = comp_seqs[(indices + self._index_offset).view(-1)]
        self._completed_sequences = st_comp_seqs.view(
            self._batch_size, self._beam_size, -1)
