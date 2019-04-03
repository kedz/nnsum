from ..parameterized import Parameterized
from ..hparam_registry import HParams

from nnsum.seq2seq.search_state import SearchState

import re


@Parameterized.register_object("seq2seq.wikibio_postprocessor")
class WikiBioPostProcessor(Parameterized):
    
    hparams = HParams()

    @hparams()
    def vocab(self):
        pass

    def init_object(self):
        self._eos_pat = " ?" + self.vocab.stop_token

    def __call__(self, inputs, search_state):
        if isinstance(search_state, SearchState):
            outputs = search_state["output"]
            attn = search_state["context_attention"]
        else:
            outputs = search_state.get_result("output")
            attn = search_state.get_result("context_attention")

        # Only write the top search item if search returns multiple candidates.
        if outputs.dim() == 3:
            outputs = outputs[:,:,0]
            attn = attn[:,:,0,:]

        outputs = outputs.t().tolist()
        texts = []
        for i, output in enumerate(outputs):
            tokens = [self.vocab[idx] for idx in output 
                      if idx != self.vocab.pad_index]
            for j, idx in enumerate(output):
                if idx == self.vocab.unknown_index:
                    copy_token = inputs["copy_sequence"][i][
                        attn[j,i,:len(tokens)].argmax()]
                    tokens[j] = copy_token
                         
            text = " ".join(tokens)
            text = re.sub(self._eos_pat, "", text)
            texts.append(text)

        return texts
