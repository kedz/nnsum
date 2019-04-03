from ..parameterized import Parameterized
from ..hparam_registry import HParams

import sys
import json
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path


@Parameterized.register_object("metrics.sequence_eval_script")
class SequenceEvalScript(Parameterized):
 
    hparams = HParams()

    @hparams(default=False)
    def debug(self):
        pass
    
    @hparams()
    def script_path(self):
        pass

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

        with NamedTemporaryFile("w") as ref_fp,\
                NamedTemporaryFile("w") as hyp_fp:
            for refs, hyp in zip(self._references, self._hypotheses):
                print(hyp, file=hyp_fp, flush=True)
                print("\n".join(refs), end="\n\n", file=ref_fp, flush=True)

            script_path = Path(self.script_path).resolve()
            
            proc_run = subprocess.run(
                ["bash", script_path, ref_fp.name, hyp_fp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            output = proc_run.stdout.decode("utf8")
            if self.debug:
                print("{} stdout:".format(script_path))
                print(output, file=sys.stderr)
                print("{} stderr:".format(script_path))
                print(proc_run.stderr.decode("utf8"), file=sys.stderr)
            
            results = json.loads(output)
            self._cache = results

        return results

    def pretty_print(self):
        if self._cache is None:
            results = self.compute()
        else:
            results = self._cache
        print(results)
