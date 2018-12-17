import ujson as json

import torch
from .summarization_dataset import SummarizationDataset


class SampleCacheDataset(SummarizationDataset):
    def __init__(self, vocab, inputs_dir, targets_dir=None, 
                 references_dir=None, sentence_limit=None, shuffle_sents=False,
                 num_samples=25, temperature=.05):
        super(SampleCacheDataset, self).__init__(
            vocab, inputs_dir, targets_dir=targets_dir,
            references_dir=references_dir, sentence_limit=sentence_limit,
            shuffle_sents=shuffle_sents)
        self.num_samples = num_samples
        self.temperature = temperature

    def _read_targets(self, raw_inputs_data, inputs_data, perm=None):
        path = self._get_targets_path(raw_inputs_data, inputs_data)
        raw_targets_data = json.loads(path.read_text())
        assert raw_targets_data["id"] == raw_inputs_data["id"]
        
        label_scores = sorted(
            raw_targets_data["label_scores"], key=lambda x: x["score"])
        
        doc_size = inputs_data["num_sentences"]
        scores = []
        labels = []
        for label_score in label_scores[-self.num_samples:]:
            scores.append(label_score["score"])
            lsize = len(label_score["labels"])
            if lsize < doc_size:
                labels.append(
                    label_score["labels"] + [0] * (doc_size - lsize))
            else:
                labels.append(label_score["labels"][:doc_size])

        labels = torch.LongTensor(labels)    
        scores = torch.FloatTensor(scores) / self.temperature
        scores = torch.softmax(scores, 0)
        assert labels.size(1) == inputs_data["sentence_lengths"].size(0)
        
        if perm is not None:
            labels = labels[:,perm]

        return {"samples": labels, "scores": scores}
