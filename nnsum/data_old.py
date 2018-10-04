import torch
from torch.utils.data import Dataset, DataLoader

import ujson as json
from collections import defaultdict


class SingleDocumentDataset(Dataset):

    class SingleDocumentItem(object):
        def __init__(self, id, document, targets, document_length,
                     sentence_lengths, reference_paths, 
                     sentence_texts, pretty_sentence_lengths):
            self.id = id
            self.document = document
            self.targets = targets
            self.document_length = document_length
            self.sentence_lengths = sentence_lengths
            self.reference_paths = reference_paths
            self.sentence_texts = sentence_texts
            self.pretty_sentence_lengths = pretty_sentence_lengths

    class SingleDocumentBatch(object):
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
    
    def __init__(self, vocab, inputs_dir, labels_dir=None, references_dir=None,
                 sentence_limit=None):

        self._vocab = vocab
        self._inputs = [path for path in inputs_dir.glob("*.json")]
        self._inputs.sort()

        self._sentence_limit = sentence_limit

        if labels_dir:
            self._labels = [path for path in labels_dir.glob("*.json")]
            self._labels.sort()
            assert len(self._labels) == len(self._inputs)
        else:
            self._labels = None

        if references_dir:
            self._reference_paths = defaultdict(list)
            for path in references_dir.glob("*"):
                doc_id = path.stem.rsplit(".", 1)[0]
                self._reference_paths[doc_id].append(path)
        else:
            self._reference_paths = None

    @property
    def vocab(self):
        return self._vocab

    @property
    def sentence_limit(self):
        return self._sentence_limit

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index):

        with self._inputs[index].open("r") as fp:
            example = json.loads(fp.read())
        
        doc_size = len(example["inputs"])
        if self.sentence_limit:
            doc_size = min(self.sentence_limit, doc_size)

        sent_size = max([len(sent["tokens"])
                         for sent in example["inputs"][:doc_size]])

        document = torch.LongTensor(doc_size, sent_size).fill_(0)
        sentence_sizes = [] 
        for s, sent in enumerate(example["inputs"][:doc_size]):
            sentence_sizes.append(len(sent["tokens"]))
            for t, token in enumerate(sent["tokens"]):
                document[s, t] = self.vocab.index(token.lower())
        sentence_sizes = torch.LongTensor(sentence_sizes)                
        sentence_texts = [sent["text"] 
                          for sent in example["inputs"][:doc_size]]
        pretty_sentence_lengths = torch.LongTensor(
            [len(sent["text"].split()) 
             for sent in example["inputs"][:doc_size]])

        if self._labels:
            with self._labels[index].open("r") as fp:
                labels = json.loads(fp.read())
            assert labels["id"] == example["id"]
            targets = torch.LongTensor(labels["labels"][:doc_size])
        else:
            targets = None

        if self._reference_paths:
            reference_paths = self._reference_paths[example["id"]]
            if len(reference_paths) == 0:
                raise Exception("Document {} has no references!".format(
                    example["id"]))
        else:
            reference_paths = None

        return self.SingleDocumentItem(
            example["id"], document, targets, doc_size, sentence_sizes,
            reference_paths, sentence_texts, pretty_sentence_lengths)

    def dataloader(self, batch_size=16, shuffle=True, num_workers=8):
        def collate_fn(batch):
            
            document_length = torch.LongTensor(
                [item.document_length for item in batch])
            document_length, indices = torch.sort(
                document_length, descending=True)

            batch_size = document_length.size(0)
            max_doc = document_length.max()
            max_sent = max([item.sentence_lengths.max() for item in batch])

            sentence_lengths = torch.LongTensor(batch_size, max_doc)
            sentence_lengths.fill_(0)
            
            documents = torch.LongTensor(batch_size, max_doc, max_sent)
            documents.fill_(0)
            
            if self._labels:
                targets = torch.LongTensor(batch_size, max_doc).fill_(-1)
            else:
                targets = None

            if self._reference_paths:
                reference_paths = []
            else:
                reference_paths = None

            ids = []
            sentence_texts = []
            pretty_sentence_lengths = torch.LongTensor(
                batch_size, max_doc).fill_(0)

            for b, index in enumerate(indices):
                item = batch[index]
                ids.append(item.id)
                doc_size, sent_size = item.document.size()
                
                sentence_lengths[b, :doc_size].copy_(
                    item.sentence_lengths)
                documents[b, :doc_size, :sent_size].copy_(item.document)
                sentence_texts.append(item.sentence_texts)
                pretty_sentence_lengths[b, :doc_size].copy_(
                    item.pretty_sentence_lengths)

                if self._labels:
                    targets[b, :doc_size].copy_(item.targets)
                if self._reference_paths:
                    reference_paths.append(item.reference_paths)

            batch = self.SingleDocumentBatch(
                ids, documents, targets, document_length, sentence_lengths,
                reference_paths, sentence_texts, pretty_sentence_lengths)
            return batch

        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate_fn)
