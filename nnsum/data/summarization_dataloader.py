import torch
from torch.utils.data import DataLoader
from nnsum.util import batch_pad_and_stack_matrix, batch_pad_and_stack_vector


class SummarizationDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, 
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(SummarizationDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, 
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)

    class SummarizationBatch(object):
        def __init__(self, id, document, targets, num_sentences,
                     sentence_lengths, reference_paths,
                     sentence_texts, pretty_sentence_lengths):
            self.id = id
            self.document = document
            self.targets = targets
            self.num_sentences = num_sentences
            self.sentence_lengths = sentence_lengths
            self.reference_paths = reference_paths
            self.sentence_texts = sentence_texts
            self.pretty_sentence_lengths = pretty_sentence_lengths

        def to(self, device=-1):
            if device < 0:
                return self
            else:
                document = self.document.to(device)
                if self.targets is not None:
                    targets = self.targets.to(device)
                else:
                    targets = None
                num_sentences = self.num_sentences.to(device)
                sentence_lengths = self.sentence_lengths.to(device)
                return self.__class__(self.id, document, targets,
                                           num_sentences, sentence_lengths,
                                           self.reference_paths, 
                                           self.sentence_texts,
                                           self.pretty_sentence_lengths)
 
    


    def _collate_fn(self, batch):

        pad = self.dataset.vocab.pad_index

        batch.sort(key=lambda x: x["num_sentences"], reverse=True)
        ids = [item["id"] for item in batch]
        documents = batch_pad_and_stack_matrix(
            [item["document"] for item in batch], pad)
        
        num_sentences = torch.LongTensor(
            [item["num_sentences"] for item in batch])
        sentence_lengths = batch_pad_and_stack_vector(
            [item["sentence_lengths"] for item in batch], 0)
        
        sentence_texts = [item["pretty_sentences"] for item in batch]
        pretty_sentence_lengths = [item["pretty_sentence_lengths"]
                                   for item in batch]

        if "targets" in batch[0]:
            targets = batch_pad_and_stack_vector(
                [item["targets"] for item in batch], -1)
        else:
            targets = None

        if "reference_paths" in batch[0]:
            reference_paths = [item["reference_paths"] for item in batch]
        else:
            reference_paths = None

        return self.SummarizationBatch(
            ids, documents, targets, num_sentences,
            sentence_lengths, reference_paths,
            sentence_texts, pretty_sentence_lengths)
