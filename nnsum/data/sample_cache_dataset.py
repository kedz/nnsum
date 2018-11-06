import ujson as json

import torch
from .summarization_dataset import SummarizationDataset


class SampleCacheDataset(SummarizationDataset):
    def __init__(self, vocab, inputs_dir, targets_dir=None, 
                 references_dir=None, sentence_limit=None,
                 num_samples=25, temperature=.05, band=[0.4,0.6]):
        super(SampleCacheDataset, self).__init__(
            vocab, inputs_dir, targets_dir=targets_dir,
            references_dir=references_dir, sentence_limit=sentence_limit)
        self.num_samples = num_samples
        self.temperature = temperature
        self.band = band

    def _read_targets(self, raw_inputs_data, inputs_data):
        path = self._get_targets_path(raw_inputs_data, inputs_data)
        raw_targets_data = json.loads(path.read_text())
        assert raw_targets_data["id"] == raw_inputs_data["id"]
        
        label_scores = sorted(
            raw_targets_data["label_scores"], key=lambda x: x["score"])
        
        doc_size = inputs_data["num_sentences"]
        scores = []
        labels = []
        for label_score in label_scores[-self.num_samples:]:
            if (label_score["score"] < self.band[0] or label_score["score"] > self.band[1]) and len(labels)>0:
              continue
            scores.append(label_score["score"])
            lsize = len(label_score["labels"])
            if lsize < doc_size:
                labels.append(
                    label_score["labels"] + [0] * (doc_size - lsize))
            else:
                labels.append(label_score["labels"][:doc_size])

        if len(labels)>1:
          del labels[1]
          del scores[1]
        assert(len(labels)>0)
        labels = torch.LongTensor(labels)    
        scores = torch.FloatTensor(scores) / self.temperature
        scores = torch.softmax(scores, 0)
        assert labels.size(1) == inputs_data["sentence_lengths"].size(0)

        return {"samples": labels, "scores": scores}
