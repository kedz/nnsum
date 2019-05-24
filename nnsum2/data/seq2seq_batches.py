from ..parameterized import Parameterized
from ..hparam_registry import HParams
from .parallel_dataset import ParallelDataset

import torch
from torch.utils.data import DataLoader
from nnsum.data.seq2seq_batcher import (
    batch_source, batch_target, batch_pointer_data,
)
from . import batch_utils


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

    @hparams(default=None, required=False)
    def control_vocabs(self):
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

    @hparams(default=None, required=False)
    def copy_field(self):
        pass

    @hparams(default=False)
    def create_extended_vocab(self):
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

    @hparams(default=None, required=False)
    def cloze_vocab_name(self):
        pass

    @hparams(default=None, required=False)
    def cloze_field(self):
        pass

    def _sort_batch(self, batch):
        for label in self.source_vocabs.keys():
            lengths = [len(example["source"]["sequence"][label])
                       for example in batch]
            I = list(range(len(lengths)))
            I.sort(key=lambda x: lengths[x], reverse=True)
            return [batch[i] for i in I]

    def _src_tgt_collate_fn(self, batch):
        if self.sort:
            batch = self._sort_batch(batch)
       
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        batch_data = batch_utils.s2s.source(sources, self.source_vocabs,
                                            max_length=self.max_source_length)
        batch_data.update(
            batch_utils.s2s.target(targets, self.target_vocabs,
                                   max_length=self.max_target_length))

        if self.copy_field:
            batch_data["copy_sequence"] = batch_utils.s2s.copy_sequence(
                sources, 
                self.copy_field,
                self.source_vocabs[self.copy_field].start_token,
                max_length=self.max_target_length)

        if self.create_extended_vocab:
            copy_field = self.copy_field
            extended_vocab = batch_utils.s2s.extend_vocab(
                sources, copy_field, self.target_vocabs[copy_field],
                max_length=self.max_source_length)
            source_extended_vocab_map = batch_utils.map_tokens(
                [item["sequence"] for item in sources], 
                copy_field, extended_vocab, start_token=True,
                max_length=self.max_source_length)
            copy_targets = batch_utils.map_tokens(
                [item["sequence"] for item in targets], 
                copy_field, extended_vocab, stop_token=True,
                max_length=self.max_target_length)

            batch_data["extended_vocab"] = extended_vocab
            batch_data["source_extended_vocab_map"] = source_extended_vocab_map
            batch_data["copy_targets"] = copy_targets

        if len(self.control_vocabs) > 0:
            batch_data["controls"] = batch_utils.s2s.discrete_controls(
                sources, self.control_vocabs)

        if self.cloze_vocab_name is not None:
            batch_data.update(
                batch_utils.fg.cloze_batch2(
                    sources,
                    self.source_vocabs[self.cloze_vocab_name],
                    self.cloze_vocab_name,
                    self.cloze_field))

        return batch_data



        input()


        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)
        if self.copy_sequence is not None:
            start_token = self.source_vocabs[self.copy_sequence].start_token
            #data["copy_sequence"] = 
            
            [[start_token] + item[self.copy_sequence] 
                                     for item in source_items]

        target_items = [item["target"]["sequence"] for item in batch]
        data.update(batch_target(target_items, self.target_vocabs))

        if "reference_string" in batch[0]["target"]:
            tgt_ref_strs = [[item["target"]["reference_string"]]
                            for item in batch]
            data["target_reference_strings"] = tgt_ref_strs

        if len(self.control_vocabs) > 0:
            ctrl_data = {}
            for ctrl, ctrl_vocab in self.control_vocabs.items():
                ctrls = torch.LongTensor(
                    [ctrl_vocab[item["source"]["controls"][ctrl]]
                     for item in batch])
                ctrl_data[ctrl] = ctrls
            data["controls"] = ctrl_data
            
        if self.create_extended_vocab:
            copy_field = self.copy_sequence
            extended_vocab = batch_utils.s2s.extended_vocab(
                source_items, copy_field, self.target_vocabs[copy_field])
            source_extended_vocab_map = batch_utils.map_tokens(
                source_items, copy_field, extended_vocab, start_token=True)

            data["extended_vocab"] = extended_vocab
            data["source_extended_vocab_map"] = source_extended_vocab_map

            data.update(
                batch_pointer_data(source_items, self.target_vocabs,
                                   targets=target_items,
                                   sparse=True))

        return data

    def _src_tgt_multiref_fn(self, batch):
        if self.sort:
            batch = self._sort_batch(batch)

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)
        if self.copy_sequence is not None:
            start_token = self.source_vocabs[self.copy_sequence].start_token
            data["copy_sequence"] = [[start_token] + items[self.copy_sequence] 
                                     for item in source_items]

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

        if len(self.control_vocabs) > 0:
            ctrl_data = {}
            for ctrl, ctrl_vocab in self.control_vocabs.items():
                ctrls = torch.LongTensor(
                    [ctrl_vocab[item["source"]["controls"][ctrl]]
                     for item in batch])
                ctrl_data[ctrl] = ctrls
            data["controls"] = ctrl_data
      
        if self.create_extended_vocab:
            raise Exception(
                "Create extended vocab for multiref not implemented.")
            data.update(
                batch_pointer_data(source_items, self.target_vocabs))

        return data

    def init_object(self):
        if self.control_vocabs is None:
            self._control_vocabs = {}

        if self.multireference:
            collate_fn = self._src_tgt_multiref_fn
        else:
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

        if "copy_sequence" in batch:
            new_batch["copy_sequence"] = batch["copy_sequence"]

        if "controls" in batch:
            new_batch["controls"] = {
                ctrl: tensor.cuda(self.device)
                for ctrl, tensor in batch["controls"].items()
            }
        if "extended_vocab" in batch:
            new_batch["extended_vocab"] = batch["extended_vocab"]
            new_batch["source_extended_vocab_map"] = \
                batch["source_extended_vocab_map"].cuda(self.device)
        if "copy_targets" in batch:
            new_batch["copy_targets"] = \
                batch["copy_targets"].cuda(self.device)
        if batch.get("cloze_indices", None) is not None:
            new_batch["cloze_indices"] = \
                batch["cloze_indices"].cuda(self.device)
            new_batch["cloze_targets"] = \
                batch["cloze_targets"].cuda(self.device)
            new_batch["cloze_lengths"] = \
                batch["cloze_lengths"].cuda(self.device)


        return new_batch

    def __len__(self):
        return len(self._dataloader)
