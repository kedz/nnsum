import torch
import torch.nn as nn

from collections import OrderedDict


class MultiLabelEmbeddingContext(nn.Module):
    def __init__(self, embedding_contexts):
        super(MultiLabelEmbeddingContext, self).__init__()
        self.embedding_contexts = nn.ModuleList(embedding_contexts)  
    
#?    @property
#?    def output_size(self):
#?        if self._merge_mode == "concat":
#?            return sum([ec.output_size 
#?                        for ec in self.embedding_contexts])
#?        else:
#?            for ec in self.embedding_contexts:
#?                return ec.output_size

    @property
    def vocab(self):
        vocabs = OrderedDict()
        for ec in self.embedding_contexts:
            vocabs[ec.name] = ec.vocab
        return vocabs

    @property
    def named_vocabs(self):
        return self.vocab

    def label_frequencies(self):
        freqs = OrderedDict()
        for ec in self.embedding_contexts:
            freqs[ec.name] = ec.label_frequencies()[ec.name]
        return freqs

    def forward(self, inputs):
        
        output = OrderedDict()
        if isinstance(inputs, (list, tuple)):
            for i, ec in enumerate(self.embedding_contexts):
                output[ec.name] = ec(inputs[i])
        elif isinstance(inputs, dict):
            for ec in self.embedding_contexts:
                output[ec.name] = ec(inputs[ec.name])
            embs = [ec(inputs[ec.name]) for ec in self.embedding_contexts]
        elif inputs.dim() == 2:
            for ec in self.embedding_contexts:
                output[ec.name] = ec(inputs)
                    
        elif inputs.dim() == 3:
            for i, ec in enumerate(self.embedding_contexts):
                output[ec.name] = ec(inputs[i])
        else:
            raise Exception(
                "Input must be a list, tuple, or dict, 2dd or 3d tensor.")

        return output

    def initialize_parameters(self):
        for ec in self.embedding_contexts:
            ec.initialize_parameters()
