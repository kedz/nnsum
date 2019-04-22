from ..parameterized import Parameterized
from ..hparam_registry import HParams


@Parameterized.register_object("binarizing_feature")
class BinarizingFeature(Parameterized):

    hparams = HParams()

    @hparams(default=0)
    def threshold(self):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, value):
        if not isinstance(value, (int, float)):
            raise Exception("Expecting numerical values, int or float.")
        if value > self.threshold:
            return 1
        else:
            return 0

    def __iter__(self):
        yield self.threshold
        yield self.threshold + 1

    @property
    def pad_index(self):
        return None
