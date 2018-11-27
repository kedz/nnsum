import torch
import torch.nn as nn

from collections import OrderedDict


class MultiEmbeddingContext(nn.Module):
    def __init__(self, embedding_contexts, merge='concat'):
        super(MultiEmbeddingContext, self).__init__()
    
        assert merge in ["concat", "sum", "mean"]

        self.embedding_contexts = nn.ModuleList(embedding_contexts)  
        self._merge_mode = merge
    
    @property
    def output_size(self):
        if self._merge_mode == "concat":
            return sum([ec.output_size 
                        for ec in self.embedding_contexts])
        else:
            for ec in self.embedding_contexts:
                return ec.output_size

    @property
    def vocab(self):
        vocabs = OrderedDict()
        for ec in self.embedding_contexts:
            vocabs[ec.name] = ec.vocab
        return vocabs

    @property
    def named_vocabs(self):
        return self.vocab

    def forward(self, inputs):

        if isinstance(inputs, (list, tuple)):
            embs = [ec(inp) 
                    for ec, inp in zip(self.embedding_contexts, inputs)]
        elif isinstance(inputs, dict):
            embs = [ec(inputs[ec.name]) for ec in self.embedding_contexts]
        else:
            raise Exception("Input must be a list, tuple, or dict.")

        if self._merge_mode == "concat":
            embs = torch.cat(embs, dim=2) 
        elif self._merge_mode == "sum":
            embs = sum(embs)
        elif self._merge_mode == "mean":
            embs = sum(embs) / len(embs)

        return embs
