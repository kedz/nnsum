import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("seq2clf.fg_cloze_model")
class FGClozeModel(Module):

    hparams = hparam_registry()

    @hparams()
    def source_embedding_context(self):
        pass
    
    @hparams()
    def target_embedding_context(self):
        pass

    @hparams()
    def source_encoder(self):
        pass

    @hparams()
    def target_encoder(self):
        pass

    @hparams()
    def attention_mechanism(self):
        pass    

    @hparams()
    def output_embedding_context(self):
        pass

    def forward(self, inputs):
        print(inputs)
        src_embs = self.source_embedding_context(inputs["source_features"])
        src_hiddens, _ = self.source_encoder(src_embs)
        cloze_index = src_embs.size(0) // 2 + 1
        src_hidden = src_hiddens[cloze_index:cloze_index + 1]

        tgt_embs = self.source_embedding_context(inputs["target_features"])
        tgt_hiddens, _ = self.target_encoder(tgt_embs)
        attention, _, tgt_read = self.attention_mechanism(
            tgt_hiddens.permute(1, 0, 2), src_hidden, 
            context_mask=inputs["target_mask"])
        tgt_read = tgt_read.squeeze(0)
        target_logits = self.output_embedding_context(tgt_read)
        return {"attention": attention, "target_logits": target_logits}

    def initialize_parameters(self):
        self.source_embedding_context.initialize_parameters()
        self.target_embedding_context.initialize_parameters()
        self.source_encoder.initialize_parameters()
        self.target_encoder.initialize_parameters()
        self.attention_mechanism.initialize_parameters()
        self.output_embedding_context.initialize_parameters()
