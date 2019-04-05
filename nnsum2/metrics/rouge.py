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


        with rouge_papier.util.TempFileManager() as manager:
            path_data = []
            for hyp, refs in zip(self._hypotheses, self._references):

                    #summary = "\n".join(text)
                hyp_path = manager.create_temp_file(hyp)
                ref_paths = [manager.create_temp_file(ref)
                             for ref in refs]
                path_data.append((hyp_path, ref_paths))

            config_text = rouge_papier.util.make_simple_config_text(path_data)
            config_path = manager.create_temp_file(config_text)
            df = rouge_papier.compute_rouge(
                config_path, max_ngram=2, lcs=True,
                remove_stopwords=False,
                length=100)
        self._cache = df.loc["average"].to_dict()

        return self._cache

    def pretty_print(self):
        if self._cache is None:
            results = self.compute()
        else:
            results = self._cache
        print(results)
