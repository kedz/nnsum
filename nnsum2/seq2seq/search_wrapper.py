from ..parameterized import Parameterized
from ..hparam_registry import HParams
from nnsum.seq2seq.greedy_search import GreedySearch
from nnsum.seq2seq.beam_search import BeamSearch


@Parameterized.register_object("seq2seq.search_wrapper")
class SearchWrapper(Parameterized):

    hparams = HParams()
 
    @hparams(default=1)
    def beam_size(self):
        pass

    @hparams(default=100)
    def max_steps(self):
        pass

    @hparams(default=True)
    def return_incomplete(self):
        pass

    def __call__(self, model, inputs, encoded_inputs=None):

        if self.beam_size == 1:
            return self._greedy_decode(model, inputs, 
                                       encoded_inputs=encoded_inputs)
        else:
            return self._beam_decode(model, inputs, 
                                     encoded_inputs=encoded_inputs) 

    def _beam_decode(self, model, inputs, encoded_inputs=None):

        if encoded_inputs is None:
            encoded_inputs = model.encode(inputs)

        context, encoder_state = encoded_inputs
        
        return BeamSearch(model.decoder, encoder_state, context, 
                          max_steps=self.max_steps, 
                          beam_size=self.beam_size,
                          return_incomplete=self.return_incomplete)

    def _greedy_decode(self, model, inputs, encoded_inputs=None):

        if encoded_inputs is None:
            encoded_inputs = model.encode(inputs)

        context, encoder_state = encoded_inputs

        return GreedySearch(model.decoder, encoder_state, context, 
                            max_steps=self.max_steps,
                            return_incomplete=self.return_incomplete)
