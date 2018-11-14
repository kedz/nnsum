import torch
import torch.nn as nn


class MultiEmbeddingContext(nn.Module):
    def __init__(self, embedding_contexts, merge='concat'):
        super(MultiEmbeddingContext, self).__init__()
    
        for key in embedding_contexts:
            assert isinstance(embedding_contexts[key], nn.Module)

        assert merge in ["concat", "sum", "mean"]

        self.embedding_contexts = nn.ModuleDict(embedding_contexts)  
        self._merge_mode = merge
        self._ordered_contexts = sorted(embedding_contexts.keys())
    
    @property
    def output_size(self):
        if self._merge_mode == "concat":
            return sum([ec.embedding_size 
                        for ec in self.embedding_contexts.values()])
        else:
            for ec in self.embedding_contexts.values():
                return ec.embedding_size

    def forward(self, input):

        embs = [self.embedding_contexts[feat](input[feat].t())
                for feat in self._ordered_contexts]
        if self._merge_mode == "concat":
            embs = torch.cat(embs, dim=2) 
        elif self._merge_mode == "sum":
            embs = sum(embs)
        elif self._merge_mode == "mean":
            embs = sum(embs) / len(embs)

        return embs
