import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from ..attention import (NoAttention, BiLinearSoftmaxAttention, 
#                         BiLinearSigmoidAttention)
from ..attention import MultiHeadAttention

class TransformerSentenceExtractor(nn.Module):
    def __init__(self, input_size, transformer_layers=6, 
                 attention_heads=10, attention_head_size=25,
                 max_position=1000, dropout=0.75): 

        super(TransformerSentenceExtractor, self).__init__()

        self.teacher_forcing = True

        self.position_embeddings = nn.Parameter(
            torch.FloatTensor(1, max_position, input_size).normal_())

        self.layer_norm = nn.LayerNorm(input_size)

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(input_size, num_heads=attention_heads,
                                head_size=attention_head_size) 
             for l in range(transformer_layers)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_size, input_size) 
             for l in range(transformer_layers)])
        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(input_size) 
             for l in range(transformer_layers)])
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(input_size)
             for l in range(transformer_layers)])

        self.dropout = dropout
        self.output_layer = nn.Linear(input_size, 1)


    def forward(self, sentence_embeddings, num_sentences, targets=None):

        input = sentence_embeddings + self.position_embeddings[
                :,:sentence_embeddings.size(1)]
        input = self.layer_norm(input)

        all_scores = []
        for attn, lin, norm1, norm2 in zip(
                self.attention_layers, self.linear_layers, 
                self.layer_norms1, self.layer_norms2):
            output, scores = attn(
                input, input, input,
                num_sentences)
            all_scores.append(scores)
            input = norm1(output + input)

            output = lin(input)
            input = F.dropout(
                norm2(output + input), p=self.dropout, training=self.training)

        logits = self.output_layer(input).squeeze(2)

        return logits, all_scores

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(
                " TransformerSentenceExtractor initialization started.")
        for name, p in self.named_parameters():
            if "layer_norm" in name:
                if logger:
                    logger.info(" {} ({}): random normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.normal_(p)    
            elif "weight" in name:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal(p)    
            elif "bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant(p, 0)    
            else:
                if logger:
                    logger.info(" {} ({}): random normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.normal_(p)    
        if logger:
            logger.info(
                " TransformerSentenceExtractor initialization finished.")
