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

    def word_counts(self):
        if isinstance(self[0]["sequence"], (list, tuple)):
            return self._word_counts_from_list()
        else:
            return self._word_counts_from_dict()
    
    def _word_counts_from_list(self):
        counts = {}
        for ex in self:
            for token in ex["sequence"]:
                counts[token] = counts.get(token, 0) + 1
        return {"tokens": counts}

    def _word_counts_from_dict(self):
        features = set(self[0]["sequence"].keys())
        counts = {ftr: dict() for ftr in features}
        for ex in self:
            if set(ex["sequence"].keys()) != features:
                raise Exception("Expected features: {} but found: {}".format(
                    str(features), str(set(ex["sequence"].keys()))))
            for ftr, tokens in ex["sequence"].items():
                for token in tokens:
                    counts[ftr][token] = counts[ftr].get(token, 0) + 1
        return counts            
