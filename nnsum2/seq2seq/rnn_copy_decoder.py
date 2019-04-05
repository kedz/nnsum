import torch
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.search_state import SearchState
from nnsum.seq2seq.rnn_state import RNNState


@register_module("seq2seq.rnn_copy_decoder")
class RNNCopyDecoder(Module):

    hparams = hparam_registry()

    @hparams()
    def input_embedding_context(self):
        pass

    @hparams()
    def rnn(self):
        pass

    @hparams()
    def context_attention(self):
        pass

    @hparams()
    def output_embedding_context(self):
        pass

    @hparams()
    def copy_switch(self):
        pass

    def _embed_inputs(self, in_feats):
        
        # Replace input features from the extended vocabulary 
        # with the unknown token. in_feats can be both a dict of matrices 
        # or a single matrix. Replacement should only be necessary when
        # running the a search, i.e. not during training.
        # NOTE consider flagging this to only happen on self.training == False.
        unk_idx = self.input_embedding_context.vocab.unknown_index
        if isinstance(in_feats, dict):
            in_vocabs = self.input_embedding_context.named_vocabs
            in_feats = {
                ftr: in_feats[ftr].masked_fill(
                    in_feats[ftr].ge(len(vcb)), unk_idx)
                for ftr, vcb in in_vocabs.items()
            }
        else:
            in_vocab = self.input_embedding_context.vocab
            in_feats = in_feats.masked_fill(
                in_feats.ge(len(in_vocab)), unk_idx)

        # Lookup input feature embeddings.
        return self.input_embedding_context(in_feats)

    def _project_attention(self, attns, vmaps):
        if isinstance(vmaps, list):
            return self._project_attention_sparse(attns, vmaps)
        else:
            return self._project_attention_dense(attns, vmaps)

    def _project_attention_sparse(self, attns, vmaps):
        mapped_attns = [] 
        for attn, vmap in zip(attns.split(1, dim=1), vmaps):
            #X = attn.squeeze(1).matmul(vmap)
            #mapped_attn = torch.mm(vmap.t().clone().detach().data, attn.squeeze(1).t()).t()\
            x = vmap.t().detach()
            x.requires_grad = False
            print(type(x))
            print(vmap)
            # probs is steps x 1 x ext_vocab_size
            mapped_attn = x.mm(attn.squeeze(1).t()).t()\
                .unsqueeze(1)
            #mapped_attn = vmap.t().matmul(attn.squeeze(1).t()).t().unsqueeze(1)
            print(mapped_attn.size())
            mapped_attn.sum().backward()
            print()
            input()
            mapped_attns.append(mapped_attn)
        mapped_attns = torch.cat(mapped_attns, dim=1)
        return mapped_attns

    def _project_attention_dense(self, attns, vmaps):
        return attns.permute(1, 0, 2).bmm(vmaps).permute(1, 0, 2)

    def forward(self, rnn_state, input_features, context, attention_state):

        rnn_input = self._embed_inputs(input_features)
        if isinstance(rnn_state, RNNState):
            rnn_state = [rnn_state[0], rnn_state[1]]
        rnn_output, rnn_state = self.rnn(rnn_input, state=rnn_state)

        ctx_attn, ctx_attn_state, ctx_emb = self.context_attention(
            context["encoder_output"], 
            rnn_output, 
            context_mask=context.get("source_mask", None),
            attention_state=attention_state)
            
        hidden_state = torch.cat([rnn_output, ctx_emb], 2)
        gen_logits = self.output_embedding_context(hidden_state) 
        gen_prob = torch.softmax(gen_logits, dim=2)

        copy_switch = self.copy_switch(torch.cat([hidden_state, rnn_input], 2))
        gen_switch = (1. - copy_switch)

        copy_prob = self._project_attention(
            ctx_attn, context["source_vocab_map"])
        
        pointer_probability = copy_switch * copy_prob
        generator_probability = gen_switch * gen_prob

        return SearchState(rnn_input=rnn_input, 
                           rnn_output=rnn_output,
                           rnn_state=RNNState.new_state(rnn_state),
                           context_attention=ctx_attn,
                           context_attention_state=ctx_attn_state,
                           attention_output=ctx_emb,
                           target_logits=gen_logits,
                           copy_switch=copy_switch.squeeze(2),
                           pointer_probability=pointer_probability,
                           generator_probability=generator_probability)
            
    def next_state(self, prev_state, context, compute_log_probability=False,
                   compute_output=False, compute_top_k=-1):

        ctx_attn_state = prev_state.get("context_attention_state", None)

        next_state = self.forward(
            prev_state["rnn_state"], 
            prev_state["output"].t(),
            context,
            ctx_attn_state)

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
            output_prob.data.masked_fill_(mask, 1e-12)
            output_log_prob = torch.log(output_prob)
            #output_log_prob.data.masked_fill_(mask, float("-inf"))
            next_state["log_probability"] = output_log_prob

        if compute_output:
            output_lp, output = next_state["log_probability"].max(2)
            next_state["output"] = output
            next_state["output_log_probability"] = output_lp
        
        return next_state
 


        if compute_output:
            if compute_top_k > 0:
                #do top k
                if compute_log_probability:
                    #get_log_prob
                    pass
            else:
                if compute_log_probability:
                    output_lp, output = next_state["log_probability"].max(2)
                    next_state["output"] = output
                    next_state["output_log_probability"] = output_lp
                else:
                    _, output = next_state["target_logits"].max(2)
                    next_state["output"] = output

        return next_state

                          
    def initialize_parameters(self):
        self.input_embedding_context.initialize_parameters()
        self.rnn.initialize_parameters()
        self.output_embedding_context.initialize_parameters()
        self.copy_switch.initialize_parameters()

    def start_inputs(self, batch_size, device=None):
        inputs = torch.tensor(
            [[self.input_embedding_context.vocab.start_index]] * batch_size,
            device=device)
        return inputs

    #TODO update searches to not use this
    @property
    def embedding_context(self):
        return self.input_embedding_context

    def set_dropout(self, dropout):
        self.rnn.set_dropout(dropout)
        self.input_embedding_context.set_dropout(dropout)
        self.output_embedding_context.set_dropout(dropout)
