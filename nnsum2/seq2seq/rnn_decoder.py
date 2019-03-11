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
    def output_embedding_context(self):
        pass

    def forward(self, rnn_state, input_features, context):
        rnn_input = self.input_embedding_context(input_features)
        rnn_output, rnn_state = self.rnn(rnn_input, state=rnn_state)
        attention_output = None

        if attention_output is not None:
            hidden_state = torch.cat([rnn_output, attention_output], 2)
        else:
            hidden_state = rnn_output

        target_logits = self.output_embedding_context(hidden_state) 
        
        return SearchState(rnn_input=rnn_input, rnn_output=rnn_output,
                           rnn_state=RNNState.new_state(rnn_state),
                           context_attention=attention_output,
                           target_logits=target_logits)
                          
    def initialize_parameters(self):
        self.input_embedding_context.initialize_parameters()
        self.rnn.initialize_parameters()
        self.output_embedding_context.initialize_parameters()
