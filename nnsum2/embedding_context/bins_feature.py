from ..parameterized import Parameterized
from ..hparam_registry import HParams


@Parameterized.register_object("bins_feature")
class BinsFeature(Parameterized):

    hparams = HParams()

    @hparams()
    def thresholds(self):
        pass

    def __len__(self):
        return len(self.thresholds) + 1

    def __getitem__(self, value):
        if not isinstance(value, (int, float)):
            raise Exception("Expecting numerical values, int or float.")
        # this should be a binary search but I'm tired.
        bin = 0
        while bin != len(self.thresholds) and value > self.thresholds[bin]:
            bin += 1

#        if bin == 0:
#            print(value, self.thresholds[bin])
#        elif bin == len(self.thresholds):
#            print(self.thresholds[bin-1], vaulue)
#        else:
#            print(self.thresholds[bin-1], value, self.thresholds[bin])
        
        return bin

    def __iter__(self):
        for val in self.thresholds:
            yield val
        yield val + 1

    @property
    def pad_index(self):
        return None
