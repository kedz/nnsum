from ..parameterized import Parameterized
from ..hparam_registry import HParams
from sacremoses import MosesDetokenizer
from nnsum.seq2seq.greedy_search import GreedySearch
from nnsum.seq2seq.beam_search import BeamSearch
import re


@Parameterized.register_object("seq2seq.decoder_wrapper")
class DecoderWrapper(Parameterized):

    hparams = HParams()
    
    #@hparams()
    #def model(self):
    #    pass


    @hparams(default=1)
    def beam_size(self):
        pass

    @hparams(default=True)
    def remove_special_tokens(self):
        pass

    @hparams(default=True)
    def detokenize(self):
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
        

        search = BeamSearch(model.decoder, encoder_state, context, 
                            max_steps=self.max_steps, 
                            beam_size=self.beam_size,
                            return_incomplete=self.return_incomplete)
        outputs = search.get_result("output").cpu()[:,:,0].t()

        vocab = model.decoder.output_embedding_context.vocab

        texts = []
        for row in outputs.tolist():
            tokens = [vocab[idx] for idx in row 
                    if idx != vocab.pad_index]

            if self.detokenize:
                text = MosesDetokenizer().detokenize(tokens)
            else:
                text = " ".join(tokens)

            if self.remove_special_tokens:
                text = re.sub(r"<.*?>", r"", text).strip()
                text = re.sub(r'  +', r' ', text)
            texts.append(text)
            
        return texts

    def _greedy_decode(self, model, inputs, encoded_inputs=None):

        if encoded_inputs is None:
            encoded_inputs = model.encode(inputs)

        context, encoder_state = encoded_inputs

        search = GreedySearch(model.decoder, encoder_state, context, 
                              max_steps=self.max_steps,
                              return_incomplete=self.return_incomplete)
        outputs = search.get_result("output").cpu().t()

        vocab = model.decoder.output_embedding_context.vocab

        texts = []
        for row in outputs.tolist():
            tokens = [vocab[idx] for idx in row 
                    if idx != vocab.pad_index]

            if self.detokenize:
                text = MosesDetokenizer().detokenize(tokens)
            else:
                text = " ".join(tokens)

            if self.remove_special_tokens:
                text = re.sub(r"<.*?>", r"", text).strip()
                text = re.sub(r'  +', r' ', text)
            texts.append(text)
            
        return texts
