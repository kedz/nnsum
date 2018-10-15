import torch
from torch.utils.data import Dataset

import pathlib
import ujson as json


class SummarizationDataset(Dataset):
    def __init__(self, vocab, inputs_dir, targets_dir=None, 
                 references_dir=None, sentence_limit=None):

        if isinstance(inputs_dir, str):
            inputs_dir = pathlib.Path(inputs_dir)
        if targets_dir and isinstance(targets_dir, str):
            targets_dir = pathlib.Path(targets_dir)
        if references_dir and isinstance(references_dir, str):
            references_dir = pathlib.Path(references_dir)

        self._vocab = vocab
        self._sentence_limit = sentence_limit
        self._inputs = [path for path in inputs_dir.glob("*.json")]
        self._inputs.sort()

        self._targets_dir = targets_dir

        if references_dir:
            self._references_paths = self._collect_references(references_dir)
        else:
            self._references_paths = None

    @property
    def vocab(self):
        return self._vocab

    @property
    def sentence_limit(self):
        return self._sentence_limit

    @staticmethod
    def _get_references(inputs_data, ref_dir):
        refs = []
        for path in ref_dir.glob("{}*".format(inputs_data["id"])):
            ref_id = path.stem.rsplit(".")[0]
            if ref_id == inputs_data["id"]:
                refs.append(path)
        return refs

    def _collect_references(self, references_dir):
        all_ref_paths = []
        for path in self._inputs:
            inputs_data = json.loads(path.read_text())
            ref_paths = self._get_references(inputs_data, references_dir)
            if len(ref_paths) == 0:
                raise Exception(
                    "No references found for example id: {}".format(
                        inputs_data["id"]))
            all_ref_paths.append(ref_paths)
        return all_ref_paths
    
    def __len__(self):
        return len(self._inputs)

    def _read_inputs(self, data):
        
        # Get the length of the document in sentences. 
        doc_size = len(data["inputs"])
        
        # Get the token lengths of each sentence and the maximum sentence
        # sentence length. If sentence_limit is set, truncate document
        # to that length.
        if self.sentence_limit:
            doc_size = min(self.sentence_limit, doc_size)
        sent_sizes = torch.LongTensor(
            [len(sent["tokens"]) for sent in data["inputs"][:doc_size]])
        sent_size = sent_sizes.max().item()
        
        # Create a document matrix of size doc_size x sent_size. Fill in
        # the word indices with vocab.
        document = torch.LongTensor(doc_size, sent_size).fill_(
            self.vocab.pad_index)
        for s, sent in enumerate(data["inputs"][:doc_size]):
            for t, token in enumerate(sent["tokens"]):
                document[s, t] = self.vocab[token.lower()]

        # Get pretty sentences that are detokenized and their lengths for 
        # generating the actual sentences.
        pretty_sentences = [sent["text"] for sent in data["inputs"][:doc_size]]
        pretty_sentence_lengths = torch.LongTensor(
            [len(sent.split()) for sent in pretty_sentences])

        return {"id": data["id"], "num_sentences": doc_size, 
                "sentence_lengths": sent_sizes, "document": document,
                "pretty_sentences": pretty_sentences,
                "pretty_sentence_lengths": pretty_sentence_lengths}
       
    def _get_targets_path(self, raw_inputs_data, inputs_data):
        return self._targets_dir / "{}.json".format(inputs_data["id"])

    def _read_targets(self, raw_inputs_data, inputs_data):
        path = self._get_targets_path(raw_inputs_data, inputs_data)
        raw_targets_data = json.loads(path.read_text())
        assert raw_targets_data["id"] == raw_inputs_data["id"]
        
        if self.sentence_limit:
            targets = torch.LongTensor(
                raw_targets_data["labels"][:self.sentence_limit])
        else:
            targets = torch.LongTensor(raw_targets_data["labels"])
        assert targets.size(0) == inputs_data["sentence_lengths"].size(0)

        return targets
        
    def __getitem__(self, index):

        raw_inputs_data = json.loads(self._inputs[index].read_text())
        inputs_data = self._read_inputs(raw_inputs_data)
        
        if self._targets_dir:
            targets_data = self._read_targets(raw_inputs_data, inputs_data)
            inputs_data["targets"] = targets_data

        if self._references_paths:
            inputs_data["reference_paths"] = self._references_paths[index]

        return inputs_data
