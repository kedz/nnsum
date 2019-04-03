import torch
from ..module import Module, register_module, hparam_registry
from nnsum.seq2seq.search_state import SearchState
from nnsum.seq2seq.rnn_state import RNNState


@register_module("seq2seq.rnn_decoder")
class RNNDecoder(Module):

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

    def forward(self, rnn_state, input_features, context, attention_state):
        rnn_input = self.input_embedding_context(input_features)
        if isinstance(rnn_state, RNNState):
            rnn_state = [rnn_state[0], rnn_state[1]]
        rnn_output, rnn_state = self.rnn(rnn_input, state=rnn_state)

        A = self.context_attention(
            context["encoder_output"], 
            rnn_output, 
            context_mask=context.get("source_mask", None),
            attention_state=attention_state)
        context_attention, context_attention_state, attention_output  = A

        if attention_output is not None:
            hidden_state = torch.cat([rnn_output, attention_output], 2)
        else:
            hidden_state = rnn_output

        target_logits = self.output_embedding_context(hidden_state) 
        
        return SearchState(rnn_input=rnn_input, rnn_output=rnn_output,
                           rnn_state=RNNState.new_state(rnn_state),
                           context_attention=context_attention,
                           context_attention_state=context_attention_state,
                           attention_output=attention_output,
                           target_logits=target_logits)
            
    def next_state(self, prev_state, context, compute_log_probability=False,
                   compute_output=False, compute_top_k=-1):

        ctx_attn_state = prev_state.get("context_attention_state", None)

        next_state = self.forward(
            prev_state["rnn_state"], 
            prev_state["output"].t(),
            context,
            ctx_attn_state)

        if compute_log_probability:
            next_state["log_probability"] = torch.log_softmax(
                next_state["target_logits"], dim=2)

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
