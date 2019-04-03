from ..parameterized import Parameterized
from ..hparam_registry import HParams
from .parallel_dataset import ParallelDataset

import torch
from torch.utils.data import DataLoader
from nnsum.data.seq2seq_batcher import batch_source, batch_target


@Parameterized.register_object("data.fg_seq2seq_batches")
class FGSeq2SeqBatches(Parameterized):

    hparams = HParams()

    @hparams(type="submodule")
    def dataset(self):
        pass

    @hparams(default=None, required=False)
    def teacher_dataset(self):
        pass

    @hparams(default=1)
    def batch_size(self):
        pass

    @hparams(default=1)
    def teacher_batch_size(self):
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

    @hparams()
    def label_vocabs(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    @hparams(default=True)
    def sort(self):
        pass

    @hparams(default=False)
    def multireference(self):
        pass

    @device.setter
    def device(self, device):
        self._device = device

    def _src_sort_batch(self, batch):
        for label in self.source_vocabs.keys():
            lengths = [len(example["sequence"][label])
                       for example in batch]
            I = list(range(len(lengths)))
            I.sort(key=lambda x: lengths[x], reverse=True)
            return [batch[i] for i in I]

    def _src_tgt_sort_batch(self, batch):
        for label in self.source_vocabs.keys():
            lengths = [len(example["source"]["sequence"][label])
                       for example in batch]
            I = list(range(len(lengths)))
            I.sort(key=lambda x: lengths[x], reverse=True)
            return [batch[i] for i in I]

    def _master_collate_fn(self, batch):
        if "source" in batch[0]:
            if self.multireference:
                return self._src_tgt_multiref_fn(batch)
            else:
                return self._src_tgt_collate_fn(batch)
        else:
            return self._src_collate_fn(batch) 

    def _src_collate_fn(self, batch):
        if self.sort:
            batch = self._src_sort_batch(batch)
        source_items = [item["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        source_label_items = [item["labels"] for item in batch]
        label_tensors = {}
        for name, label_vocab in self.label_vocabs.items():
            labels = [label_vocab[item[name]] for item in source_label_items]
            labels = torch.LongTensor(labels)
            label_tensors[name] = labels
        data["source_labels"] = label_tensors

        #target_items = [item["sequence"] for item in batch]
        #data.update(batch_target(target_items, self.target_vocabs))

        return data

    def _src_tgt_collate_fn(self, batch):
        
        if self.sort:
            batch = self._src_tgt_sort_batch(batch)
        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        
        source_label_items = [item["source"]["labels"] for item in batch]
        label_tensors = {}
        for name, label_vocab in self.label_vocabs.items():
            labels = [label_vocab[item[name]] for item in source_label_items]
            labels = torch.LongTensor(labels)
            label_tensors[name] = labels
        data["source_labels"] = label_tensors

        target_items = [item["target"]["sequence"] for item in batch]
        data.update(batch_target(target_items, self.target_vocabs))

        if "reference_string" in batch[0]["target"]:
            tgt_ref_strs = [[item["target"]["reference_string"]]
                            for item in batch]
            data["target_reference_strings"] = tgt_ref_strs

        return data

    def _src_tgt_multiref_fn(self, batch):
        if self.sort:
            batch = self._src_tgt_sort_batch(batch)

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        source_label_items = [item["source"]["labels"] for item in batch]
        label_tensors = {}
        for name, label_vocab in self.label_vocabs.items():
            labels = [label_vocab[item[name]] for item in source_label_items]
            labels = torch.LongTensor(labels)
            label_tensors[name] = labels
        data["source_labels"] = label_tensors

        num_refs = [len(ex["target"]["references"]) for ex in batch]
        max_refs = max(num_refs)

        target_items = []
        feature, target_vocab = list(self.target_vocabs.items())[0] 

        for ex in batch:
            seqs = []
            for ref in ex["target"]["references"]:
                seqs.append(ref["sequence"])
            if len(seqs) < max_refs:
                diff = max_refs - len(seqs)
                seqs.extend([{feature: [target_vocab.pad_token]}] * diff)
            target_items.extend(seqs)

        target_data = batch_target(target_items, self.target_vocabs)
        tgt_out_ftrs = target_data["target_output_features"][feature].view(
            len(batch), max_refs, -1)
        tgt_in_ftrs = target_data["target_input_features"][feature].view(
            len(batch), max_refs, -1)
        tgt_lens = target_data["target_lengths"].view(len(batch), max_refs)

        for i, nref in enumerate(num_refs):
            tgt_out_ftrs.data[i,nref:].fill_(target_vocab.pad_index)
            tgt_in_ftrs.data[i,nref:].fill_(target_vocab.pad_index)
            tgt_lens.data[i,nref:].fill_(target_vocab.pad_index)

        target_data["num_references"] = torch.LongTensor(num_refs)
        target_data["max_references"] = max_refs

        data.update(target_data) 
 
        if batch[0]["target"]["references"][0]["reference_string"]:
            data["target_reference_strings"] = [
                [ref["reference_string"] for ref in ex["target"]["references"]]
                for ex in batch
            ]
      
        return data

    def init_object(self):
        self._dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._master_collate_fn)

        if self.teacher_dataset is not None:
            self._teacher_dataloader = DataLoader(
                self.teacher_dataset, 
                batch_size=self.teacher_batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=self._master_collate_fn)
        else:
            self._teacher_dataloader = None


    def __iter__(self):
        if self._teacher_dataloader is None:
            return self._training_iter()
        else:
            return self._training_teacher_iter()

    def _training_iter(self):
        for batch in self._dataloader:
            if self.device > -1:
                batch = self.place_batch_on_gpu(batch)
            yield batch

    def _training_teacher_iter(self):
        teacher_iter = iter(self._teacher_dataloader)
        try:
            for train_batch in self._dataloader:
                try:
                    teacher_batch = next(teacher_iter)
                except StopIteration:
                    teacher_iter = iter(self._teacher_dataloader)
                    teacher_batch = next(teacher_iter)

                if self.device > -1:
                    train_batch = self.place_batch_on_gpu(train_batch)
                    teacher_batch = self.place_batch_on_gpu(teacher_batch)
                yield train_batch, teacher_batch
        except StopIteration:
            for b in teacher_iter:
                pass

    def place_batch_on_gpu(self, batch):

        new_feats = {}
        for key, tensor in batch["source_input_features"].items():
            new_feats[key] = tensor.cuda(self.device)
        new_lengths = batch["source_lengths"].cuda(self.device)

        new_batch = {"source_input_features": new_feats,
                     "source_lengths": new_lengths}

        if "source_mask" in batch:
            source_mask = batch["source_mask"].cuda(self.device)
            new_batch["source_mask"] = source_mask

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

        if "target_reference_strings" in batch:
            tgt_ref_strs = batch["target_reference_strings"]
            new_batch["target_reference_strings"] = tgt_ref_strs

        if "num_references" in batch:
            num_refs = batch["num_references"].cuda(self.device)
            new_batch["num_references"] = num_refs
            new_batch["max_references"] = batch["max_references"]

        new_batch["source_labels"] = {
            name: batch["source_labels"][name].cuda(self.device)
            for name in self.label_vocabs.keys()
        }

        return new_batch

    def __len__(self):
        return len(self._dataloader)
