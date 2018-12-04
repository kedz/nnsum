import torch
from torch.utils.data import DataLoader
from nnsum.util import batch_pad_and_stack_vector
import ujson as json
from .aligned_dataset import AlignedDataset


class Seq2ClfDataLoader(DataLoader):
    def __init__(self, dataset, source_vocabs, target_vocabs,
                 batch_size=1, shuffle=False, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, 
                 drop_last=False, timeout=0, worker_init_fn=None,
                 include_original_data=False, sorted=True):

        if isinstance(dataset, AlignedDataset):
            collate_fn = self._aligned_collate_fn 
        else:
            collate_fn = self._source_collate_fn
        super(Seq2ClfDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, 
            worker_init_fn=worker_init_fn, collate_fn=collate_fn)

        self.include_original_data = include_original_data
        self._source_vocabs = source_vocabs
        self._target_vocabs = target_vocabs
        self._sorted = sorted

    def _source_collate_fn(self, batch):
        raise Exception("Implement source only loader.")
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

        return batch_data

    def _aligned_collate_fn(self, batch):
        if self._sorted:
            batch.sort(key=lambda x: len(x[0]["tokens"]["tokens"]), 
                       reverse=True)

        targets = [[vocab[ex[1]["labels"][cls]] if cls in ex[1]["labels"] \
                        else -1
                    for cls, vocab in self._target_vocabs.items()]
                   for ex in batch]
        source_lengths = torch.LongTensor([len(ex[0]["tokens"]["tokens"]) + 2
                                           for ex in batch])
        batch_source_features = {}
        for feat, vocab in self._source_vocabs.items():
            src_feature_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] + \
                    [vocab[f] for f in ex[0]["tokens"][feat]] + \
                    [vocab.stop_index])
                src_feature_sequences.append(ftr_seq)
            
            src_feature_sequences = batch_pad_and_stack_vector(
                src_feature_sequences, vocab.pad_index)
            batch_source_features[feat] = src_feature_sequences

        batch_data = {"source_features": batch_source_features,
                      "source_lengths": source_lengths,
                      "targets": torch.LongTensor(targets)}
        
        if self.include_original_data:
            batch_data["orig_data"] = batch



        return batch_data
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
                    [vocab.start_index] + [vocab[f] for f in ex[0]["tokens"][feat]])
                src_feature_sequences.append(ftr_seq)
            
            src_feature_sequences = batch_pad_and_stack_vector(
                src_feature_sequences, vocab.pad_index)
            batch_source_features[feat] = src_feature_sequences
            
        for feat, vocab in self._target_vocabs.items():
            tgt_input_sequences = []
            tgt_output_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor(
                    [vocab.start_index] + [vocab[f] for f in ex[1]["tokens"][feat]] \
                    + [vocab.stop_index])
                tgt_input_sequences.append(ftr_seq[:-1])
                tgt_output_sequences.append(ftr_seq[1:])

            tgt_input_sequences = batch_pad_and_stack_vector(
                tgt_input_sequences, vocab.pad_index)
            tgt_output_sequences = batch_pad_and_stack_vector(
                tgt_output_sequences, vocab.pad_index)
            batch_target_input_features[feat] = tgt_input_sequences
            batch_target_output_features[feat] = tgt_output_sequences

        return batch_data
#        print(batch_data["source_lengths"].view(1, -1))
#        for name, features in batch_data["source_features"].items():
#            print(name)
#            print(features)
#        
#        exit()      

    def _collate_fn(self, batch):
        



        batch = [json.loads(item) for item in batch]
        batch.sort(key=lambda x: len(x["tokens"]), reverse=True)

        source_lengths = torch.LongTensor([len(ex["tokens"]) for ex in batch])
        batch_source_features = {}
        batch_data = {"source_features": batch_source_features,
                      "source_lengths": source_lengths}

        for feat, vocab in self._source_vocabs.items():
            feature_sequences = []
            for ex in batch:
                ftr_seq = torch.LongTensor([vocab[f] for f in ex[feat]])
                feature_sequences.append(ftr_seq)
            
            feature_sequences = batch_pad_and_stack_vector(
                feature_sequences, vocab.pad_index)
            batch_source_features[feat] = feature_sequences

        #for name, features in batch_data["source_features"].items():
        #    print(name)
        #    print(features)
              
        return batch_data  
