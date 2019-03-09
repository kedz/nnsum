from torch.utils.data import Dataset

import pathlib
import ujson as json

from ..parameterized import Parameterized
from ..hparam_registry import HParams


@Parameterized.register_object("data.parallel_dataset")
class ParallelDataset(Dataset, Parameterized):

    hparams = HParams()

    @hparams(type="submodule")
    def source(self):
        pass

    @hparams(type="submodule")
    def target(self):
        pass

    def init_object(self):
        if len(self.source) != len(self.target):
            raise Exception("source and target must have the same size.")

    def __getitem__(self, index):
        return {"source": self.source[index], "target": self.target[index]}
    
    def __len__(self):
        return len(self.source) 
