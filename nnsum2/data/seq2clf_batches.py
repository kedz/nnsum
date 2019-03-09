from ..parameterized import Parameterized
from ..hparam_registry import HParams
from .parallel_dataset import ParallelDataset

import torch
from torch.utils.data import DataLoader
from nnsum.data.seq2clf_batcher import batch_source


@Parameterized.register_object("data.seq2clf_batches")
class Seq2ClfBatches(Parameterized):

    hparams = HParams()

    @hparams(type="submodule")
    def dataset(self):
        pass

    @hparams(default=1)
    def batch_size(self):
        pass

    @hparams(default=True)
    def shuffle(self):
        pass

    @hparams(default=0)
    def num_workers(self):
        pass

    @hparams()
    def input_vocabs(self):
        pass

    @hparams()
    def label_vocabs(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    @device.setter
    def device(self, device):
        self._device = device

    def _src_tgt_collate_fn(self, batch):

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.input_vocabs)

        for name, tgt_vcb in self.label_vocabs.items():
            labels = [tgt_vcb[item["target"]["labels"][name]] 
                      for item in batch]
            break
        labels = torch.LongTensor(labels)
        data["target_labels"] = labels
        return data

    def init_object(self):
        self._dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._src_tgt_collate_fn)

    def __iter__(self):
        for batch in self._dataloader:
            if self.device > -1:
                batch = self.place_batch_on_gpu(batch)
            yield batch

    def place_batch_on_gpu(self, batch):

        new_feats = {}
        for key, tensor in batch["source_input_features"].items():
            new_feats[key] = tensor.cuda(self.device)
        new_lengths = batch["source_lengths"].cuda(self.device)

        new_batch = {"source_input_features": new_feats,
                     "source_lengths": new_lengths}

        if "source_mask" in batch:
            new_mask = batch["source_mask"].cuda(self.device)
            new_batch["source_mask"] = new_mask

        if "target_labels" in batch:
            new_labels =  batch["target_labels"].cuda(self.device)
            new_batch["target_labels"] = new_labels

        return new_batch

    def __len__(self):
        return len(self._dataloader)
