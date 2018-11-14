import torch
from torch.utils.data import Dataset

import pathlib
import ujson as json


class RAMDataset(Dataset):
    def __init__(self, data_path):
        if not isinstance(data_path, pathlib.Path):
            data_path = pathlib.Path(data_path)
        if data_path.is_dir():
            self._data = [json.loads(path.read_text()) 
                          for path in data_path.glob("*") 
                          if not path.name.startswith(".")]
            self._data.sort(key=lambda x: x["id"]) 
        else:
            with data_path.open("r") as fp:
                self._data = [json.loads(line) for line in fp]

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data) 
