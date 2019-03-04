import torch
from torch.utils.data import DataLoader
from nnsum.util import batch_pad_and_stack_vector
import numpy as np
import ujson as json
from .aligned_dataset import AlignedDataset

from collections import OrderedDict
from .seq2clf_batcher import batch_source


class Seq2ClfDataLoader(DataLoader):
    def __init__(self, dataset, source_vocabs, target_vocabs,
                 batch_size=1, shuffle=False, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, 
                 drop_last=False, timeout=0,
                 include_original_data=False, sorted=True):

        super(Seq2ClfDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, 
            collate_fn=self._collate_fn)

        self.include_original_data = include_original_data
        self._source_vocabs = source_vocabs
        self._target_vocabs = target_vocabs
        self._sorted = sorted

    @property
    def source_vocabs(self):
        return self._source_vocabs

    @property
    def target_vocabs(self):
        return self._target_vocabs

    def _sort_batch(self, batch):
        key = list(self.source_vocabs.keys())[0]
        indices = np.argsort([-len(item["source"]["sequence"][key]) 
                              for item in batch])
        return [batch[i] for i in indices]
 
    def _collate_fn(self, batch):
        if self._sorted:
            batch = self._sort_batch(batch)

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        for name, tgt_vcb in self.target_vocabs.items():
            labels = [tgt_vcb[item["target"]["labels"][name]] 
                      for item in batch]
            break
        labels = torch.LongTensor(labels)
        data["target_labels"] = labels
        return data
#    def _source_collate_fn(self, batch):
#        raise Exception("Implement source only loader.")
#        if self._sorted:
#            batch.sort(key=lambda x: len(x["tokens"]["tokens"]), reverse=True)
#
#        lengths = torch.LongTensor([len(ex["tokens"]["tokens"]) for ex in batch])
#
#
#        source_lengths = lengths.add_(1)
#
#        batch_source_features = {}
#        batch_data = {"source_features": batch_source_features,
#                      "source_lengths": source_lengths}
#
#        if self.include_original_data:
#            batch_data["orig_data"] = [ex for ex in batch]
#
#        for feat, vocab in self._source_vocabs.items():
#            src_feature_sequences = []
#            for ex in batch:
#                ftr_seq = torch.LongTensor(
#                    [vocab.start_index] + [vocab[f] for f in ex["tokens"][feat]])
#                src_feature_sequences.append(ftr_seq)
#            
#            src_feature_sequences = batch_pad_and_stack_vector(
#                src_feature_sequences, vocab.pad_index)
#            batch_source_features[feat] = src_feature_sequences
#
#        return batch_data
#
#    def _aligned_collate_fn(self, batch):
#        if self._sorted:
#            batch.sort(key=lambda x: len(x[0]["tokens"]["tokens"]), 
#                       reverse=True)
#
#        targets = OrderedDict()
#
#        for cls, vocab in self._target_vocabs.items():
#            labels = [] 
#            for ex in batch:
#                if cls in ex[1]["labels"]:
#                    lbl = vocab[ex[1]["labels"][cls]] 
#                else:
#                    lbl = vocab["(n/a)"]
#                labels.append(lbl)
#            targets[cls] = torch.LongTensor(labels)
#        source_lengths = torch.LongTensor([len(ex[0]["tokens"]["tokens"]) + 2
#                                           for ex in batch])
#        batch_source_features = {}
#        source_mask = None
#        for feat, vocab in self._source_vocabs.items():
#            src_feature_sequences = []
#            for ex in batch:
#                ftr_seq = torch.LongTensor(
#                    [vocab.start_index] + \
#                    [vocab[f] for f in ex[0]["tokens"][feat]] + \
#                    [vocab.stop_index])
#                src_feature_sequences.append(ftr_seq)
#            
#            src_feature_sequences = batch_pad_and_stack_vector(
#                src_feature_sequences, vocab.pad_index)
#            if source_mask is None:
#                source_mask = src_feature_sequences.eq(vocab.pad_index)
#
#            batch_source_features[feat] = src_feature_sequences
#
#        
#
#        batch_data = {"source_features": batch_source_features,
#                      "source_lengths": source_lengths,
#                      "source_mask": source_mask,
#                      "targets": targets}
#        
#        if self.include_original_data:
#            batch_data["orig_data"] = batch
#
#        return batch_data
