import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_state import SearchState
from .rnn_decoder import RNNDecoder


class PointerGeneratorDecoder(RNNDecoder):
    def __init__(self, embedding_context, hidden_dim=512, num_layers=1,
                 rnn_cell="GRU", attention="dot", dropout=0.):
        super(PointerGeneratorDecoder, self).__init__(
            embedding_context, hidden_dim=hidden_dim, num_layers=num_layers,
            rnn_cell=rnn_cell, attention=attention, dropout=0.)

        self._input_switch_net = nn.Linear(embedding_context.output_size, 1)
        self._state_switch_net = nn.Linear(hidden_dim, 1, bias=False)
        self._context_switch_net = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, prev_rnn_state, inputs, context):
        inputs = self._mask_extended_vocabulary(inputs)
        next_state = super(PointerGeneratorDecoder, self).forward(
            prev_rnn_state, inputs, context)

        copy_attention = next_state["context_attention"]
        target_distribution = torch.softmax(next_state["target_logits"], dim=2)

        source_probability = self._compute_source_probability(
            next_state["rnn_input"],
            next_state["rnn_output"],
            next_state["weighted_context"])
        target_probability = 1. - source_probability

        source_distribution = []

        source_distribution = self._project_attention(
            copy_attention, context["source_vocab_map"])

#        print(copy_attention.permute(1, 0, 2).size())
#        attn_steps = copy_attention.permute(1, 0, 2).split(1, dim=1)
#        print(attn_steps[0].size())
#        print(context["source_vocab_map"].size())
#        input()
#        for attn_step in attn_steps:
#            
#            source_distribution.append(
#                attn_step.bmm(context["source_vocab_map"]).permute(1, 0, 2))
#        source_distribution = torch.cat(source_distribution, 0)

        pointer_probability = source_probability * source_distribution
        generator_probability = target_probability * target_distribution
       
        next_state["copy_attention"] = copy_attention
        next_state["source_probability"] = source_probability.squeeze(-1)
        next_state["pointer_probability"] = pointer_probability
        next_state["target_probability"] = target_probability.squeeze(-1)
        next_state["generator_probability"] = generator_probability

        return next_state

    def _project_attention(self, attns, vmaps):
        if isinstance(vmaps, list):
            return self._project_attention_sparse(attns, vmaps)
        else:
            return self._project_attention_dense(attns, vmaps)

    def _project_attention_sparse(self, attns, vmaps):
        mapped_attns = [] 
        for attn, vmap in zip(attns.split(1, dim=1), vmaps):
            # probs is steps x 1 x ext_vocab_size
            mapped_attn = vmap.t().matmul(attn.squeeze(1).t()).t().unsqueeze(1)
            mapped_attns.append(mapped_attn)
        mapped_attns = torch.cat(mapped_attns, dim=1)
        return mapped_attns

    def _project_attention_dense(self, attns, vmaps):
        return attns.permute(1, 0, 2).bmm(vmaps).permute(1, 0, 2)

    def next_state(self, prev_state, context, compute_log_probability=False,
                   compute_output=False):
        next_state = self.forward(
            prev_state["rnn_state"], 
            prev_state["output"].t(),
            context)

        if compute_log_probability or compute_output:
            extended_vsize = next_state["pointer_probability"].size(2)
            target_vsize = next_state["generator_probability"].size(2)
            diff = extended_vsize - target_vsize
            if diff > 0:
                pad = (0, diff, 0, 0, 0, 0)
                padded_gen_probability = F.pad(
                    next_state["generator_probability"], pad, "constant", 0)
            else:
                padded_gen_probability = next_state["generator_probability"]
            output_prob = (
                next_state["pointer_probability"] + padded_gen_probability
            )
            mask = output_prob.eq(0)
            output_prob.data.masked_fill_(mask, 1)
            output_log_prob = torch.log(output_prob)
            output_log_prob.data.masked_fill_(mask, float("-inf"))
            next_state["log_probability"] = output_log_prob

        if compute_output:
            output_lp, output = next_state["log_probability"].max(2)
            next_state["output"] = output
            next_state["output_log_probability"] = output_lp
        
        return next_state
    
    def _mask_extended_vocabulary(self, inputs):
        # TODO make inputs an aggregate collection object.
        unk_idx = self.embedding_context.vocab.unknown_index
        vsize = len(self.embedding_context.vocab)
        if isinstance(inputs, dict): 
            for name in inputs.keys():
                input = inputs[name]
                inputs[name] = input.masked_fill(input.data.ge(vsize), unk_idx)
            return inputs
        elif isinstance(inputs, list):
            return [input.masked_fill(input.data.ge(vsize), unk_idx)
                    for input in inputs]
        else:
            return inputs.masked_fill(inputs.data.ge(vsize), unk_idx)

    def _compute_source_probability(self, inputs, rnn_output, context):
        a = self._input_switch_net(inputs)
        b = self._state_switch_net(rnn_output)
        c = self._context_switch_net(context)
        return torch.sigmoid(a + b + c)

    def initialize_parameters(self):
        super(PointerGeneratorDecoder, self).initialize_parameters()
