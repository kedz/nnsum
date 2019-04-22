import torch
import torch.nn.functional as F

from torch_scatter import scatter_add

from nnsum2.module import Module, register_module, hparam_registry
from nnsum2.seq2seq.searches import SearchState


@register_module("seq2seq.decoders.rnn_pointer_generator")
class RNNPointerGenerator(Module):

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
        
        #print(self.input_embedding_context.named_vocabs["toks"].size())
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

    def _map_attention_to_vocab(self, attention, source_indices, vocab_size):
        tgt_steps, batch_size, src_size = attention.size()
        return scatter_add(
            attention,
            source_indices.unsqueeze(0).repeat(tgt_steps, 1, 1),
            dim_size=vocab_size)

    def forward(self, rnn_state, input_features, context, attention_state,
                compute_output=False, compute_log_probability=False):

        rnn_input = self._embed_inputs(input_features)
        rnn_output, rnn_state = self.rnn(rnn_input, state=rnn_state)

        c_attn, c_attn_state, c_value = self.context_attention(
            context.get("encoder_output", None),
            rnn_output, 
            key_mask=context.get("source_mask", None),
            state=attention_state)
            
        hidden_state = torch.cat([rnn_output, c_value], 2)
        gen_logits = self.output_embedding_context(hidden_state) 
        gen_dist = torch.softmax(gen_logits, dim=2)

        copy_switch = self.copy_switch(torch.cat([hidden_state, rnn_input], 2))
        gen_switch = (1. - copy_switch)

        copy_dist = self._map_attention_to_vocab(
            c_attn, context["source_extended_vocab_map"],
            len(context["extended_vocab"]))

        ptr_prob = copy_switch * copy_dist
        gen_prob = gen_switch * gen_dist

        if c_attn_state is not None:
            c_attn_state = (c_attn_state, ("batch", "sequence", "support"))

        if c_attn is not None:
            c_attn = (c_attn, ("sequence", "batch", "support"))

        if c_value is not None:
            c_value = (c_value, ("sequence", "batch", "embedding"))

        state = SearchState(
            rnn_input=(rnn_input, ("sequence", "batch", "embedding")),
            rnn_output=(rnn_output, ("sequence", "batch", "embedding")),
            decoder_state=(rnn_state, ("layers", "batch", "embedding")),
            context_attention=c_attn,
            context_attention_state=c_attn_state,
            attention_output=c_value,
            copy_switch=(copy_switch.squeeze(2), ("sequence", "batch")),
            generator_logits=(gen_logits, ("sequence", "batch", "vocab")),
            generator_distribution=(
                gen_dist, ("sequence", "batch", "vocab")
            ),
            pointer_distribution=(
                copy_dist, ("sequence", "batch", "ext_vocab")
            ),
            generator_probability=(
                gen_prob, ("sequence", "batch", "vocab")
            ),
            pointer_probability=(
                ptr_prob, ("sequence", "batch", "ext_vocab")
            ),
        )

        if compute_log_probability or compute_output:

            seq_size, batch_size, vsize = gen_prob.size() 
            vocab_index = torch.arange(vsize, device=gen_prob.device).view(
                1, 1, -1).repeat(seq_size, batch_size, 1)
            out_prob = scatter_add(gen_prob, vocab_index, out=ptr_prob.clone())

            mask = out_prob.eq(0)
            out_prob.data.masked_fill_(mask, 1e-12)
            out_log_prob = torch.log(out_prob)
            #output_log_prob.data.masked_fill_(mask, float("-inf"))
            #output_log_prob.data.masked_fill_(mask, float("-inf"))
            state["log_probability"] = (
                out_log_prob, ("sequence", "batch", "ext_vocab")
            )

        if compute_output:
            state["output"] = (
                out_log_prob.argmax(2).t(),
                ("batch", "sequence")
            )

        return state

    def next_state(self, prev_state, context, compute_log_probability=False,
                   compute_output=False):

        next_state = self.forward(
            prev_state["decoder_state"], 
            prev_state["output"],
            context,
            prev_state.get("context_attention_state", None),
            compute_log_probability=compute_log_probability,
            compute_output=compute_output)

        return next_state
                         
    def initialize_parameters(self):
        self.input_embedding_context.initialize_parameters()
        self.rnn.initialize_parameters()
        self.output_embedding_context.initialize_parameters()
        self.copy_switch.initialize_parameters()
        self.context_attention.initialize_parameters()

    def set_dropout(self, dropout):
        self.rnn.set_dropout(dropout)
        self.input_embedding_context.set_dropout(dropout)
        self.context_attention.set_dropout(dropout)
        self.copy_switch.set_dropout(dropout)
        self.output_embedding_context.set_dropout(dropout)

    def initialize_search_state(self, init_state):
        batch_size = init_state["encoder_state"].size(1)
        output = init_state["encoder_state"].new().long().new(batch_size, 1)
        output.fill_(self.input_embedding_context.vocab.start_index)
        return SearchState(
            decoder_state=(
                init_state["encoder_state"],
                ("layers", "batch", "embedding"),
            ),
            output=(output, ("batch", "sequence")))
                          
    def initialize_context(self, init_context):
        return init_context 
