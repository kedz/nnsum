import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class EmbeddingContext(nn.Module):
    
    @staticmethod
    def update_command_line_options(parser):
        parser.add_argument("--embedding-size", type=int, required=True)
        parser.add_argument(
            "--pretrained-embeddings", type=str, required=False,
            default=None)
        parser.add_argument("--top-k", type=int, required=False, default=None)
        parser.add_argument("--at-least", type=int, required=False, default=1)
        parser.add_argument(
            "--word-dropout", type=float, required=False, default=0.0)
        parser.add_argument(
            "--embedding-dropout", type=float, required=False, default=0.0)
        parser.add_argument(
            "--update-rule", type=str, required=False, default="update-all",
            choices=["update-all", "fix-all"],)
        parser.add_argument(
            "--filter-pretrained", action="store_true", default=False)

    def __init__(self, vocab, embedding_size, word_dropout=0.0,
                 embedding_dropout=0.0, initializer=None, 
                 update_rule="update-all"):
        super(EmbeddingContext, self).__init__()

        self.embeddings = nn.Embedding(
            len(vocab), embedding_size, padding_idx=vocab.pad_index)

        self._vocab = vocab
        self._word_dropout = word_dropout
        self._embedding_dropout = embedding_dropout
        self._embedding_size = embedding_size
        self._update_rule = update_rule

        if initializer is not None:
            if (len(vocab) != initializer.size(0)) \
                    or (embedding_size != initializer.size(1)) \
                    or initializer.dim() != 2:
                raise Exception(
                    ("EmbbedingLayer: initializer has dims: ({}) "
                     " but expected ({}, {}).").format(
                         ",".join([str(s) for s in initializer.size()]),
                         len(vocab), embedding_size))
        self.initializer = initializer

        if update_rule == "fix-all":
            self.embeddings.weight.requires_grad = False


    @staticmethod
    def argparser():
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        parser.add_argument("--embedding-size", type=int, default=200,
                            required=False)
        parser.add_argument(
            "--pretrained-embeddings", type=str, required=False,
            default=None)
        parser.add_argument("--top-k", type=int, required=False, default=None)
        parser.add_argument("--at-least", type=int, required=False, default=1)
        parser.add_argument(
            "--word-dropout", type=float, required=False, default=0.0)
        parser.add_argument(
            "--embedding-dropout", type=float, required=False, default=0.25)
        parser.add_argument(
            "--update-rule", type=str, required=False, default="fix-all",
            choices=["update-all", "fix-all"],)
        parser.add_argument(
            "--filter-pretrained", action="store_true", default=False)
        return parser

    @property
    def vocab(self):
        return self._vocab
   
    @property
    def word_dropout(self):
        return self._word_dropout
   
    @property
    def embedding_dropout(self):
        return self._embedding_dropout

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def update_rule(self):
        return self._update_rule

    def apply_token_dropout(self, inputs, drop_prob):
        probs = inputs.data.new().resize_(inputs.size()[:-1]).fill_(drop_prob) 
        mask = torch.bernoulli(probs).byte().unsqueeze(-1)
        inputs.data.masked_fill_(mask, 0)
        return inputs

    def forward(self, inputs):

        if inputs.dim() == 2:
            emb = self.embeddings(inputs)
        else:
            bs = inputs.size(0)
            ss = inputs.size(1)
            ts = inputs.size(2)
            inputs_flat = inputs.view(bs * ss, ts)
            emb = self.embeddings(inputs_flat).view(bs, ss, ts, -1)
        
        emb = F.dropout(emb, p=self.embedding_dropout, training=self.training)
        if self.word_dropout > 0:
            emb = self.apply_token_dropout(emb, self.word_dropout)

        return emb

    def parameters(self):
        for p in self.embeddings.parameters():
            if p.requires_grad:
                yield p

    def named_parameters(self, memo, submod_prefix):
        for n, p in self.embeddings.named_parameters(memo, submod_prefix):
            if p.requires_grad:
                yield n, p

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" EmbeddingContext initialization started.")
        if self.initializer is not None:
            if logger:
                logger.info(" Initializing with pretrained embeddings.")
            self.embeddings.weight.data.copy_(self.initializer)
        else:
            if logger:
                logger.info(" Initializing with random normal.")
            nn.init.normal_(self.embeddings.weight)    
        if logger:
            logger.info(" EmbeddingContext initialization finished.")
