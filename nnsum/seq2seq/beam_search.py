import torch
import torch.nn.functional as F
import numpy as np

from .search_algo import DecoderSearch
from .search_state import SearchState
from .rnn_state import RNNState


class BeamSearch(DecoderSearch):
    def __init__(self, decoder, encoder_state, context, context_mask=None,
                 beam_size=8, max_steps=1000, rescoring_func=None,
                 return_incomplete=False, sort_by_score=True, 
                 start_input=None):

        self._start_input = start_input
        self._beam_size = beam_size
        if rescoring_func is not None:
            self._rescore_beam = rescoring_func

        super(BeamSearch, self).__init__(decoder, encoder_state, context, 
                                         max_steps=max_steps,
                                         context_mask=context_mask,
                                         return_incomplete=return_incomplete)
 
        # Create structures for storing completed beam items during search
        self._completed_sequences = [list() for x in range(self.batch_size)]
        self._completed_lengths = torch.LongTensor(
            self.batch_size, beam_size).fill_(0)
        self._completed_scores = encoder_state.new(self.batch_size, beam_size)\
            .fill_(float("-inf"))

        self._index_offset = torch.arange(
            0, self.batch_size, device=encoder_state.device).view(-1, 1)
        self._index_offset.mul_(self.beam_size)
        self._beam_indices = torch.arange(self.beam_size)\
                .repeat(self.batch_size).view(1,-1)

        self._is_sorted = False
        self._sort_by_score = sort_by_score

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def is_sorted(self):
        return self._is_sorted

    @property
    def sort_by_score(self):
        return self._sort_by_score

    def _initialize_search_state(self, encoder_state):
        
        nl, bh_sz, state_dims = encoder_state.size()
        bm_sz = self.beam_size
        
        output = self.decoder.start_inputs(bh_sz * bm_sz, 
                                           device=encoder_state.device).t()
        if self._start_input is not None:
            import warnings
            warnings.warn("Using start input is experimental and will change.")
            output = self._start_input
        
        slp = encoder_state.new(bh_sz, bm_sz, 1).fill_(0.)
        slp.data[:,1:].fill_(float("-inf"))

        init_enc_state = encoder_state.unsqueeze(2).repeat(1, 1, bm_sz, 1)\
            .view(nl, bh_sz * bm_sz, state_dims)

        return SearchState(output=output, 
                           rnn_state=init_enc_state,
                           sequence_log_probability=slp)

    def _initialize_context(self, context, context_mask):
        if context is None:
            return None, None

        bh_sz, ctx_len, ctx_dims = context["encoder_output"].size()
        bm_sz = self.beam_size
        init_context = context["encoder_output"].unsqueeze(1)\
            .repeat(1, bm_sz, 1, 1)\
            .view(bh_sz * bm_sz, ctx_len, ctx_dims)

        beam_context = {"encoder_output": init_context}

        if "source_mask" in context:
            init_mask = context["source_mask"].unsqueeze(1)\
                .repeat(1, bm_sz, 1)\
                .view(bh_sz * bm_sz, ctx_len)
            beam_context["source_mask"] = init_mask

        if context.get("controls", None) is not None:
            beam_ctrl = {}
            for ctrl, ctrl_data in context["controls"].items():
                beam_ctrl[ctrl] = ctrl_data.view(-1, 1).repeat(1, bm_sz)\
                    .view(bh_sz * bm_sz)
            beam_context["controls"] = beam_ctrl
        if "source_vocab_map" in context:
            _, xsz, ysz = context["source_vocab_map"].size()
            svm_beam = context["source_vocab_map"].view(bh_sz, 1, xsz, ysz)\
                .repeat(1, bm_sz, 1, 1)\
                .view(bh_sz * bm_sz, xsz, ysz)
            beam_context["source_vocab_map"] = svm_beam

        return beam_context, None

    # TODO Make sure this interface is general enough for lm or 
    # arbitrary model based rescoring ##, history, next_tokens):
    def _rescore_beam(self, log_probs):
        return log_probs / self.steps

    def next_state(self, prev_state, active_batches):

        next_state = self.decoder.next_state(
            prev_state, self.context,
            compute_log_probability=True)

        log_probs = next_state["log_probability"].view(
            self.batch_size, self._beam_size, -1)

        topk_lps, candidate_outputs = torch.topk(
            log_probs, k=self._beam_size, dim=2)

        candidate_log_probs = prev_state["sequence_log_probability"] + topk_lps
        scores = self._rescore_beam(candidate_log_probs)

        next_scores, top_score_idxs = torch.topk(
            scores.view(self._batch_size, -1), 
            k=self._beam_size,
            dim=1)

        output_log_probability = topk_lps.view(1, self.batch_size, -1).gather(
            2, top_score_idxs.unsqueeze(0))

        output = candidate_outputs\
            .view(self.batch_size, self._beam_size ** 2)\
            .gather(1, top_score_idxs).view(-1, 1)
        next_state["output"] = output
        next_log_probs = candidate_log_probs\
            .view(self.batch_size, self._beam_size ** 2)\
            .gather(1, top_score_idxs).view(self.batch_size, self.beam_size, 1)
        next_state["sequence_log_probability"] = next_log_probs

        beam_selections = top_score_idxs / self.beam_size
        next_rnn_indxs = (beam_selections + self._index_offset).view(-1)
        if isinstance(next_state["rnn_state"], RNNState):
            next_rnn_state = next_state["rnn_state"].reindex[:, next_rnn_indxs]
            
        else:
            next_rnn_state = next_state["rnn_state"][:, next_rnn_indxs]
        
        intermediate_state = next_state
        next_state = SearchState(
            output=output.t(),
            output_log_probability=output_log_probability,
            rnn_state=next_rnn_state, 
            rnn_output=intermediate_state["rnn_output"],
            target_logits=intermediate_state["target_logits"],
            log_probability=intermediate_state["log_probability"],
            sequence_log_probability=next_log_probs,
            scores=next_scores.view(1, self.batch_size, self.beam_size))

        if "context_attention" in intermediate_state:
            next_state["context_attention"] = intermediate_state["context_attention"]
        if intermediate_state.get("context_attention_state", None) is not None:
            
            next_state["context_attention_state"] = intermediate_state[
                "context_attention_state"][next_rnn_indxs]

        if self.steps == 1:
            self._beam_history = self._beam_indices.view(
                -1, self.batch_size, self.beam_size)
        else:
            next_history = self._beam_history\
                .view(-1, self.batch_size * self.beam_size)[:,next_rnn_indxs]
            self._beam_history = torch.cat(
                [next_history, self._beam_indices], 0)
            self._beam_history = self._beam_history.view(
                    -1, self.batch_size, self.beam_size) 
        return next_state

    def check_termination(self, next_state, active_items):

        next_tokens = next_state["output"].view(self.batch_size,
                                                 self.beam_size)
        is_complete = next_tokens.eq(self.stop_index).cpu()

        for i, j in zip(*np.where(is_complete.data.numpy())):

            num_complete = len(self._completed_sequences[i])
            if num_complete == self.beam_size:
                continue

            self._completed_sequences[i].append(self._beam_history[:,i,j])
            next_state["sequence_log_probability"][i,j].data.fill_(
                float("-inf"))
            self._completed_lengths[i, num_complete] = self.steps
            score = next_state["scores"][0, i, j]
            self._completed_scores[i, num_complete] = score

            if num_complete + 1 == self.beam_size:
                active_items[i] = 0
        return active_items

    def _add_incomplete_to_beam(self, final_search_states, active_items):

        for batch in range(self._batch_size):
            num_complete = len(self._completed_sequences[batch])
            num_remaining = self._beam_size - num_complete
            if num_remaining == 0:
                continue
            start_index = 0
            scores = final_search_states["scores"]
            logprobs = final_search_states["sequence_log_probability"]

            idx = -1
            while num_complete < self._beam_size:
                idx += 1

                # Ignore items that we finished in this step
                if logprobs[batch, idx].eq(float("-inf")):
                    continue
                self._completed_sequences[batch].append(
                    self._beam_history[:,batch,idx])
                self._completed_lengths[batch, num_complete] = self.steps
                self._completed_scores[batch, num_complete] = \
                    scores[0, batch, idx]
                num_complete += 1

        return 
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



    def _collect_search_states(self, active_items):

        if self.return_incomplete and torch.any(active_items):
            self._add_incomplete_to_beam(self._state_history[-1], active_items)

        max_len = self._completed_lengths.max().item()
        max_beam = max([len(x) for x in self._completed_sequences])
        seqs = self._completed_lengths.new(
            self.batch_size, max_beam, max_len).fill_(-1)
        for i, batch in enumerate(self._completed_sequences):
            for j, cand in enumerate(batch):
                seqs[i,j,:self._completed_lengths[i,j]].copy_(cand)
        self._completed_sequences = seqs.permute(2, 0, 1)
        self._hidden_selectors = F.pad(
            self._completed_sequences[:-1], (0, 0, 0, 0, 1, 0),
            'constant', 0)

        search_states = self._state_history[0]
        for state in self._state_history[1:]:
            search_states.append(state)
        self._state_history = search_states
        self._state_history["active_items"] = self._completed_lengths.eq(0)


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

    def get_result(self, field):
        if not self.is_finished:
            self.search()
        if self.sort_by_score and not self.is_sorted:
            self._sort_results()
        return super(BeamSearch, self).get_result(field)

    def _sort_results(self):

        sorted_scores, indices = torch.sort(self._completed_scores, dim=1,
                                            descending=True)

        indices = indices.cpu()
        seq_len, bh_sz, bm_sz = self._completed_sequences.size()
        
        selector = indices.view(1, bh_sz, bm_sz).repeat(seq_len, 1, 1)
        #if self._completed_sequences.device.type != "cpu":
        #    selector = selector.cuda(self._completed_sequences.device)

        self._completed_sequences = self._completed_sequences.gather(
            2, selector)
        if self._completed_scores.device.type != "cpu":
            self._completed_sequences = self._completed_sequences.cuda(
                self._completed_scores.device)
        
        self._hidden_selectors = F.pad(
            self._completed_sequences[:-1], (0, 0, 0, 0, 1, 0),
            'constant', 0)

        self._completed_lengths = self._completed_lengths.gather(
            1, indices)

        self._completed_scores = sorted_scores
        self._is_sorted = True

    def _collect_output(self):
        outputs = self._state_history["output"].view(
            -1, self.batch_size, self.beam_size)

        mask = self._completed_sequences.eq(-1)
        selector = self._completed_sequences.masked_fill(mask, 0)
        results = outputs.gather(2, selector)
        results.data.masked_fill_(mask, self.pad_index)
        return results

    def _collect_output_lengths(self):
        return self._completed_lengths

    def _collect_target_logits(self):
        return self._collect_field("target_logits")
        selector = self._hidden_selectors
        logits = self._state_history["logits"].view(
            self.steps, self.batch_size, self.beam_size, -1)
        vsize = logits.size(3)
        mask = self._completed_sequences.eq(-1)
        selector = selector.masked_fill(mask, 0)\
            .view(self.steps, self.batch_size, self.beam_size, 1)\
            .repeat(1, 1, 1, vsize)

        logits = logits.gather(2, selector) 
        logits.data.masked_fill_(mask.unsqueeze(-1), 0.)
        return logits

    def _collect_log_probability(self):
        return self._collect_field("log_probability")
    
    def _collect_field(self, field):
        selector = self._hidden_selectors
        result = self._state_history[field].view(
            self.steps, self.batch_size, self.beam_size, -1)
        vsize = result.size(3)
        mask = self._completed_sequences.eq(-1)
        selector = selector.masked_fill(mask, 0)\
            .view(self.steps, self.batch_size, self.beam_size, 1)\
            .repeat(1, 1, 1, vsize)

        result = result.gather(2, selector) 
        result.data.masked_fill_(mask.unsqueeze(-1), 0.)
        return result

    def _collect_output_log_probability(self):
        selector = self._completed_sequences
        mask = self._completed_sequences.eq(-1)
        olp = self._state_history["output_log_probability"]
        result = olp.gather(2, selector.masked_fill(mask, 0))
        result.data.masked_fill_(mask, 0.)
        return result

    def _collect_rnn_output(self):
        return self._collect_field("rnn_output")

    def _collect_context_attention(self):
        return self._collect_field("context_attention")

    def _collect_scores(self):
        return self._completed_scores
