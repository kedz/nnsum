import torch
from torch.utils.data import Dataset

import pathlib
import ujson as json
from collections import defaultdict


class SummarizationDataset(Dataset):
    def __init__(self, vocab, inputs_dir, targets_dir=None, 
                 references_dir=None, sentence_limit=None,
                 shuffle_sents=False):

        if isinstance(inputs_dir, str):
            inputs_dir = pathlib.Path(inputs_dir)
        if targets_dir and isinstance(targets_dir, str):
            targets_dir = pathlib.Path(targets_dir)
        if references_dir and isinstance(references_dir, str):
            references_dir = pathlib.Path(references_dir)

        self._shuffle_sents = shuffle_sents
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
    def shuffle_sents(self):
        return self._shuffle_sents
    
    @property
    def vocab(self):
        return self._vocab

    @property
    def sentence_limit(self):
        return self._sentence_limit

    def _collect_references(self, references_dir):

        ref_paths = defaultdict(list)
        for path in references_dir.glob("*"):
            ref_id = path.stem.rsplit(".", 1)[0]
            ref_paths[ref_id].append(path)

        all_ref_paths = []
        for path in self._inputs:
            rp = ref_paths[path.stem]
            if len(rp) == 0:
                raise Exception(
                    "No references found for example id: {}".format(
                        path.stem))
            all_ref_paths.append(rp)
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

    def _read_targets(self, raw_inputs_data, inputs_data, perm=None):
        path = self._get_targets_path(raw_inputs_data, inputs_data)
        raw_targets_data = json.loads(path.read_text())
        assert raw_targets_data["id"] == raw_inputs_data["id"]
        
        if self.sentence_limit:
            targets = torch.LongTensor(
                raw_targets_data["labels"][:self.sentence_limit])
        else:
            targets = torch.LongTensor(raw_targets_data["labels"])
        assert targets.size(0) == inputs_data["sentence_lengths"].size(0)
        if perm is not None:
            targets = targets[perm]

        return targets
        
    def __getitem__(self, index):

        raw_inputs_data = json.loads(self._inputs[index].read_text())
        inp_data = self._read_inputs(raw_inputs_data)

        if self.shuffle_sents:
            num_sents = inp_data["sentence_lengths"].size(0)
            perm = I = torch.randperm(num_sents)
            inp_data["sentence_lengths"] = inp_data["sentence_lengths"][I]
            inp_data["pretty_sentence_lengths"] = \
                inp_data["pretty_sentence_lengths"][I]
            inp_data["document"] = inp_data["document"][I]
            inp_data["pretty_sentences"] = \
                [inp_data["pretty_sentences"][i] for i in I.tolist()]
        else:
            perm = None


        for isent, slen, psent, pslen in zip(inp_data["document"], 
                                      inp_data["sentence_lengths"],
                                      inp_data["pretty_sentences"],
                                      inp_data["pretty_sentence_lengths"]):
            try:
                assert isent.tolist().index(0) == slen
            except ValueError:
                assert isent.size(0) == slen
        
        if self._targets_dir:
            targets_data = self._read_targets(raw_inputs_data, inp_data,
                                              perm=perm)
            inp_data["targets"] = targets_data

        if self._references_paths:
            inp_data["reference_paths"] = self._references_paths[index]

        return inp_data
