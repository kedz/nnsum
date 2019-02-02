import torch
from torch.utils.data import DataLoader

import numpy as np
import ujson as json

from nnsum.util import batch_pad_and_stack_vector
from .aligned_dataset import AlignedDataset
from .seq2seq_batcher import batch_source, batch_target, batch_pointer_data


class Seq2SeqDataLoader(DataLoader):
    def __init__(self, dataset, source_vocabs, target_vocabs=None,
                 batch_size=1, shuffle=False, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, 
                 drop_last=False, timeout=0, worker_init_fn=None,
                 include_original_data=False, sort=True,
                 has_copy_attention=False):

#        if isinstance(dataset, AlignedDataset):
        collate_fn = self._aligned_collate_fn 
        #else:
        #    collate_fn = self._source_collate_fn
        self.include_original_data = include_original_data
        self.has_copy_attention = has_copy_attention
        self._source_vocabs = source_vocabs
        self._target_vocabs = target_vocabs
        self._is_sorted = sorted

        super(Seq2SeqDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, 
            worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    
    @property
    def is_sorted(self):
        return self._is_sorted

    @property
    def source_vocabs(self):
        return self._source_vocabs

    @property
    def target_vocabs(self):
        return self._target_vocabs

    def _sort_batch(self, batch):
        indices = np.argsort([-len(item["source"]["tokens"]) 
                              for item in batch])
        return [batch[i] for i in indices]
         
    def _aligned_collate_fn(self, batch):

        if self.is_sorted:
            batch = self._sort_batch(batch)

        source_items = [item["source"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        if "target" in batch[0]:
            target_items = [item["target"] for item in batch]
            data.update(batch_target(target_items, self.target_vocabs))
            data.update(batch_pointer_data(source_items, self.target_vocabs,
                                           targets=target_items))
        
        else:
            data.update(batch_pointer_data(source_items, self.target_vocabs))

        return data
















        if "references" not in batch[0][1]:
            return self._aligned_collate_fn_single_ref(batch)
        else:
            return self._aligned_collate_fn_multi_ref(batch)














    def _source_collate_fn(self, batch):

        

        if self._sorted:
            batch.sort(key=lambda x: len(x["tokens"]["tokens"]), reverse=True)

        lengths = torch.LongTensor([len(ex["tokens"]["tokens"]) for ex in batch])
        source_lengths = lengths.add_(1)

        batch_source_features = {}
        batch_data = {"source_features": batch_source_features,
                      "source_lengths": source_lengths}

        if self.include_original_data:
            batch_data["orig_data"] = [ex for ex in batch]

        for feat, vocab in self._source_vocabs.items():
            src_feature_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] + [vocab[f] for f in ex["tokens"][feat]])
                src_feature_sequences.append(ftr_seq)
            
            src_feature_sequences = batch_pad_and_stack_vector(
                src_feature_sequences, vocab.pad_index)
            batch_source_features[feat] = src_feature_sequences
        batch_data["source_mask"] = src_feature_sequences.eq(vocab.pad_index)

        return batch_data

    def _aligned_collate_fn_multi_ref(self, batch):

        if self._sorted:
            batch.sort(key=lambda x: len(x[0]["tokens"]["tokens"]), 
                       reverse=True)

        source_lengths = torch.LongTensor(
            [len(ex[0]["tokens"]["tokens"]) for ex in batch]) + 1
        target_lengths = torch.LongTensor(
            [len(ref["tokens"]["tokens"]) for ex in batch
             for ref in ex[1]["references"]])

        batch_source_features = {}
        batch_target_input_features = {}
        batch_target_output_features = {}
        batch_data = {"source_features": batch_source_features,
                      "source_lengths": source_lengths,
                      "target_input_features": batch_target_input_features,
                      "target_output_features": batch_target_output_features,
                      "target_lengths": target_lengths}
        if self.include_original_data:
            batch_data["orig_data"] = [ex for ex in batch]

        for feat, vocab in self._source_vocabs.items():
            src_feature_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] \
                    + [vocab[f] for f in ex[0]["tokens"][feat]])
                src_feature_sequences.append(ftr_seq)
            
            src_feature_sequences = batch_pad_and_stack_vector(
                src_feature_sequences, vocab.pad_index)
            batch_source_features[feat] = src_feature_sequences
        batch_data["source_mask"] = src_feature_sequences.eq(vocab.pad_index)

        ref_source_ids = torch.LongTensor(
            [i for i, ex in enumerate(batch)
               for j in range(len(ex[1]["references"]))])
        
        num_refs = torch.LongTensor([len(ex[1]["references"]) for ex in batch])
        batch_data["target_source_ids"] = ref_source_ids
        batch_data["num_references"] = num_refs

        for feat, vocab in self._target_vocabs.items():
            tgt_input_sequences = []
            tgt_output_sequences = []
            for ex in batch:
                for ref in ex[1]["references"]:
                    ftr_seq = torch.LongTensor(
                        [vocab.start_index] \
                        + [vocab[f] for f in ref["tokens"][feat]] \
                        + [vocab.stop_index])
                    tgt_input_sequences.append(ftr_seq[:-1])
                    tgt_output_sequences.append(ftr_seq[1:])

            tgt_input_sequences = batch_pad_and_stack_vector(
                tgt_input_sequences, vocab.pad_index)
            tgt_output_sequences = batch_pad_and_stack_vector(
                tgt_output_sequences, vocab.pad_index)
            batch_target_input_features[feat] = tgt_input_sequences
            batch_target_output_features[feat] = tgt_output_sequences

        batch_data["multi_ref"] = True
        return batch_data 

    def _aligned_collate_fn_single_ref(self, batch):
        if self._sorted:
            batch.sort(key=lambda x: len(x[0]["tokens"]["tokens"]), 
                       reverse=True)

        lengths = torch.LongTensor([[len(ex[0]["tokens"]["tokens"]), 
                                     len(ex[1]["tokens"]["tokens"])]
                                    for ex in batch])
        source_lengths = lengths[:,0] + 1
        target_lengths = lengths[:,1] + 1

        batch_source_features = {}
        batch_target_input_features = {}
        batch_target_output_features = {}
        batch_data = {"source_features": batch_source_features,
                      "source_lengths": source_lengths,
                      "target_input_features": batch_target_input_features,
                      "target_output_features": batch_target_output_features,
                      "target_lengths": target_lengths}

        if self.include_original_data:
            batch_data["orig_data"] = [ex for ex in batch]

        for feat, vocab in self._source_vocabs.items():
            src_feature_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] \
                    + [vocab[f] for f in ex[0]["tokens"][feat]])
                src_feature_sequences.append(ftr_seq)
            
            src_feature_sequences = batch_pad_and_stack_vector(
                src_feature_sequences, vocab.pad_index)
            batch_source_features[feat] = src_feature_sequences
        batch_data["source_mask"] = src_feature_sequences.eq(vocab.pad_index)
            
        for feat, vocab in self._target_vocabs.items():
            tgt_input_sequences = []
            tgt_output_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] \
                    + [vocab[f] for f in ex[1]["tokens"][feat]] \
                    + [vocab.stop_index])
                tgt_input_sequences.append(ftr_seq[:-1])
                tgt_output_sequences.append(ftr_seq[1:])

            tgt_input_sequences = batch_pad_and_stack_vector(
                tgt_input_sequences, vocab.pad_index)
            tgt_output_sequences = batch_pad_and_stack_vector(
                tgt_output_sequences, vocab.pad_index)
            batch_target_input_features[feat] = tgt_input_sequences
            batch_target_output_features[feat] = tgt_output_sequences

        

        if self.has_copy_attention:
        
            batch_copy_output_features = {}
            for feat, vocab in self._target_vocabs.items():
                cpy_output_sequences = []
                for ex in batch:
                    ftr_seq = torch.LongTensor(
                        [vocab.start_index] \
                        + [vocab[f] for f in ex[0]["tokens"][feat]])
                    cpy_output_sequences.append(ftr_seq)
                cpy_output_sequences = batch_pad_and_stack_vector(
                    cpy_output_sequences, vocab.pad_index)
                batch_copy_output_features[feat] = cpy_output_sequences
            batch_data["copy_features"] = batch_copy_output_features

        batch_data["multi_ref"] = False
        return batch_data
