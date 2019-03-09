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
                 make_extended_vocab=False):

#        if isinstance(dataset, AlignedDataset):
        collate_fn = self._aligned_collate_fn 
        #else:
        #    collate_fn = self._source_collate_fn
        self.include_original_data = include_original_data
        self.make_extended_vocab = make_extended_vocab
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
        key = list(self.source_vocabs.keys())[0]
        indices = np.argsort([-len(item["source"]["sequence"][key]) 
                              for item in batch])
        return [batch[i] for i in indices]
         
    def _aligned_collate_fn(self, batch):

        if self.is_sorted:
            batch = self._sort_batch(batch)

        source_items = [item["source"]["sequence"] for item in batch]
        data = batch_source(source_items, self.source_vocabs)

        # make this general. 
        data["source_tokens"] = [item["tokens"] 
                                 for item in source_items]

        if self.include_original_data:
            data["original_data"] = batch
#[item["source"]["original_data"]
 #                                    for item in batch]
        if "target" in batch[0]:
            if "references" in batch[0]["target"]:
                return self._batch_multiref_targets(data, batch)
            else:
                return self._batch_singleref_targets(data, batch)

        if self.make_extended_vocab:
            data.update(batch_pointer_data(source_items, self.target_vocabs))
        return data

    def _batch_multiref_targets(self, data, batch):
       
        feature, target_vocab = list(self.target_vocabs.items())[0] 

        num_refs = [len(ex["target"]["references"]) for ex in batch]
        max_refs = max(num_refs)

        target_items = []

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
            data["reference_strings"] = [
                [ref["reference_string"] for ref in ex["target"]["references"]]
                for ex in batch
            ]
       
        if self.make_extended_vocab:
            source_items = []
            for ex in batch:
                source_items.extend([ex["source"]["sequence"]] * max_refs)
            ev_data = batch_pointer_data(
                source_items, self.target_vocabs, targets=target_items)

            ct = ev_data["copy_targets"].view(len(batch), max_refs, -1)
            for i, nref in enumerate(num_refs):
                ct.data[i,nref:].fill_(target_vocab.pad_index)
            
            data.update(batch_pointer_data(
                [ex["source"]["sequence"] for ex in batch], 
                self.target_vocabs))
            data["copy_targets"] = ev_data["copy_targets"]
        return data
    
    def _batch_singleref_targets(self, data, batch):
        target_items = [item["target"]["sequence"] for item in batch]
        data.update(batch_target(target_items, self.target_vocabs))
        if "reference_string" in batch[0]["target"]:
            reference_strings = [[ex["target"]["reference_string"]] 
                                 for ex in batch]
            data["reference_strings"] = reference_strings

        if self.make_extended_vocab:
            source_items = [item["source"]["sequence"] for item in batch]
            data.update(batch_pointer_data(source_items, self.target_vocabs,
                                           targets=target_items))
        
        return data

#            target_items = [item["target"]["sequence"] for item in batch]
#            data.update(batch_target(target_items, self.target_vocabs))
#            if "reference_string" in batch[0]["target"]:
#                data["reference_strings"] = [
#                    [item["target"]["reference_string"]] for item in batch]
#        else:
#            target_items = None
#
#                                           targets=target_items))
#
#        return data
















#        if "references" not in batch[0][1]:
#            return self._aligned_collate_fn_single_ref(batch)
#        else:
#            return self._aligned_collate_fn_multi_ref(batch)
#
#












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
