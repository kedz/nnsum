from torch.utils.data import Dataset

import pathlib
import ujson as json

from ..parameterized import Parameterized
from ..hparam_registry import HParams


@Parameterized.register_object("data.jsonl_ram_dataset")
class JsonlRamDataset(Dataset, Parameterized):

    hparams = HParams()

    @hparams()
    def path(self):
        pass

    def init_object(self):
        with open(self.path, "r") as fp:
            self._data = [json.loads(line) for line in fp]

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data) 
