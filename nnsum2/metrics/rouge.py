from ..parameterized import Parameterized
from ..hparam_registry import HParams

import sys
import json
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path
import rouge_papier


@Parameterized.register_object("metrics.rouge")
class ROUGE(Parameterized):
 
    hparams = HParams()

    @hparams(default=False)
    def debug(self):
        pass
    
#    @hparams()
#    def script_path(self):
#        pass

    def init_object(self):
        self.reset()

    def reset(self):
        self._hypotheses = []
        self._references = []
        self._cache = None

    def __call__(self, batch, hypotheses):
        self._hypotheses.extend(hypotheses)
        self._references.extend(batch["target_reference_strings"])

    def compute(self):
        if len(self._hypotheses) == 0:
            raise RuntimeError("Must have processed at least one batch.")


        df = rouge_papier.to_dataframe(self._hypotheses, self._references,
            ngrams=2)
        
        fscore_cols = [x for x in df.columns if x[1] == 'F-Score']
        
        d = {lbl[0]: val 
             for lbl, val in zip(fscore_cols, df[fscore_cols].values.ravel())}
        self._cache = d
        return d

    def pretty_print(self):
        if self._cache is None:
            results = self.compute()
        else:
            results = self._cache
        print(results)
