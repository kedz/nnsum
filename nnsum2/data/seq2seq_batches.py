from ..parameterized import Parameterized
from ..hparam_registry import HParams
from .parallel_dataset import ParallelDataset

import torch
from torch.utils.data import DataLoader
from nnsum.data.seq2seq_batcher import batch_source, batch_target


@Parameterized.register_object("data.seq2seq_batches")
class Seq2SeqBatches(Parameterized):

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
    def source_vocabs(self):
        pass

    @hparams()
    def target_vocabs(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    @device.setter
    def device(self, device):
        self._device = device

    def _src_tgt_collate_fn(self, batch):

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        target_items = [item["target"]["sequence"] for item in batch]
        data.update(batch_target(target_items, self.target_vocabs))

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

        if "target_input_features" in batch:
            tgt_in_feats = {}
            for key, tensor in batch["target_input_features"].items():
                tgt_in_feats[key] = tensor.cuda(self.device)
            tgt_out_feats = {}
            for key, tensor in batch["target_output_features"].items():
                tgt_out_feats[key] = tensor.cuda(self.device)
            tgt_lens = batch["target_lengths"].cuda(self.device)
            tgt_mask = batch["target_mask"].cuda(self.device)
            new_batch["target_input_features"] = tgt_in_feats
            new_batch["target_output_features"] = tgt_out_feats
            new_batch["target_lengths"] = tgt_lens
            new_batch["target_mask"] = tgt_mask

        return new_batch

    def __len__(self):
        return len(self._dataloader)
