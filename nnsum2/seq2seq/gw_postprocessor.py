from ..parameterized import Parameterized
from ..hparam_registry import HParams

from nnsum.seq2seq.search_state import SearchState

from sacremoses import MosesDetokenizer
import re


@Parameterized.register_object("seq2seq.gw_postprocessor")
class GWPostProcessor(Parameterized):
    
    hparams = HParams()

    @hparams()
    def vocab(self):
        pass

    @property
    def detokenizer(self):
        return self._detokenizer

    def init_object(self):
        self._detokenizer = MosesDetokenizer()
        self._eos_pat = " ?" + self.vocab.stop_token
        self._sent_pat = r"<sent> (.)"

    def __call__(self, batch, search_state):
        if isinstance(search_state, SearchState):
            outputs = search_state["output"]
        else:
            outputs = search_state.get_result("output", mask=True)

        # Only write the top search item if search returns multiple candidates.
        if outputs.dim() == 3:
            outputs = outputs[:,0]

        vocab = batch.get("extended_vocab", self.vocab)
        outputs = outputs.tolist()
        texts = []
        for output in outputs:
            tokens = [vocab[idx] for idx in output if idx != vocab.pad_index]
            
#            for idx in output:
#                if idx < len(self.vocab):
#                    if idx != self.vocab.pad_index:
#                        tokens.append(self.vocab[idx])
#                elif extended_vocab:
#                    ext_idx = idx - len(self.vocab)
#                    if ext_idx < len(extended_vocab):
#                        tokens.append(extended_vocab[ext_idx])
#                    else:
#                        raise Exception("BAD INDEX", idx)
#                else:
#                    raise Exception("BAD INDEX", idx)
            
            #[self.vocab[idx] for idx in output 
            #          if idx != self.vocab.pad_index]
            text = self.detokenizer.detokenize(tokens)
            text = re.sub(r"@ ", r"", text)
            text = re.sub(r" @", r"", text)
            text = re.sub(self._eos_pat, "", text)
            #text = re.sub(self._sent_pat, lambda x: x.groups()[0].upper(), 
            #              text)
            texts.append(text)

        return texts
