import torch

from nnsum2.module import Module, register_module, hparam_registry
from nnsum2.seq2seq.searches import SearchState


@register_module("seq2seq.decoders.controllable_rnn")
class ControllableRNN(Module):

    hparams = hparam_registry()

    @hparams()
    def input_embedding_context(self):
        pass

    @hparams()
    def control_embedding_context(self):
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

    def forward(self, rnn_state, input_features, context, attention_state,
                compute_output=False, compute_log_probability=False):

        rnn_input = self.input_embedding_context(input_features)

        if len(self.control_embedding_context.named_vocabs) > 0:
            ctrl_embeddings = self.control_embedding_context(
                {c: t.unsqueeze(1) for c, t in context["controls"].items()})
            ctrl_embeddings = ctrl_embeddings.repeat(rnn_input.size(0),  1, 1)
            rnn_input = torch.cat([rnn_input, ctrl_embeddings], 2)


        rnn_output, rnn_state = self.rnn(rnn_input, state=rnn_state)

        c_attn, c_attn_state, c_value = self.context_attention(
            context.get("encoder_output", None),
            rnn_output, 
            key_mask=context.get("source_mask", None),
            state=attention_state)

        if c_value is not None:
            hidden_state = torch.cat([rnn_output, c_value], 2)
        else:
            hidden_state = rnn_output

        target_logits = self.output_embedding_context(hidden_state) 

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
            target_logits=(target_logits, ("sequence", "batch", "vocab")))

        if compute_output: 
            state["output"] = (
                target_logits.argmax(2).t(), 
                ("batch", "sequence")
            )

        if compute_log_probability:
            state["log_probability"] = (
                torch.softmax(target_logits, dim=2),
                ("sequence", "batch", "vocab")
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
        self.context_attention.initialize_parameters()
        self.output_embedding_context.initialize_parameters()

    def start_inputs(self, batch_size, device=None):
        inputs = torch.tensor(
            [[self.input_embedding_context.vocab.start_index]] * batch_size,
            device=device)
        return inputs

    def set_dropout(self, dropout):
        self.rnn.set_dropout(dropout)
        self.input_embedding_context.set_dropout(dropout)
        self.context_attention.set_dropout(dropout)
        self.output_embedding_context.set_dropout(dropout)

    def initialize_search_state(self, init_state):
        batch_size = init_state["encoder_state"].size(1)
        device = init_state["encoder_state"].device
        output = init_state["encoder_state"].new().long().new(batch_size, 1) \
            .fill_(self.input_embedding_context.vocab.start_index)
        return SearchState(
            decoder_state=(
                init_state["encoder_state"], ("layers", "batch", "embedding"),
            ),
            output=(output, ("batch", "sequence")))
                          
    def initialize_context(self, init_context):
        return init_context 
