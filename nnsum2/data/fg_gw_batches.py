from ..parameterized import Parameterized
from ..hparam_registry import HParams
from .parallel_dataset import ParallelDataset

import torch
from torch.utils.data import DataLoader
from nnsum.data.seq2seq_batcher import (
    batch_source, batch_target, batch_pointer_data,
)
from . import batch_utils


@Parameterized.register_object("data.fg_gw_teacher_batches")
class FGGigaWordTeacherBatches(Parameterized):

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
    def source_vocab(self):
        pass

    @hparams()
    def target_vocab(self):
        pass

    @hparams()
    def source_field(self):
        pass

    @hparams()
    def target_field(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    @hparams(default=None, required=False)
    def max_source_length(self):
        pass

    @hparams(default=None, required=False)
    def max_target_length(self):
        pass

    @device.setter
    def device(self, device):
        self._device = device

    def _src_tgt_collate_fn(self, batch):
        
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]
        data = batch_utils.fg.word_prediction_batch(
            sources, targets, self.source_vocab, self.source_field,
            self.target_vocab, self.target_field)

        return data

    def init_object(self):
        collate_fn = self._src_tgt_collate_fn
        self._dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn)

    def __iter__(self):
        for batch in self._dataloader:
            if self.device > -1:
                batch = self.place_batch_on_gpu(batch)
            yield batch

    def place_batch_on_gpu(self, batch):
        return batch_utils.fg.to_gpu(batch, self.device)

    def __len__(self):
        return len(self._dataloader)
