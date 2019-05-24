import torch

from nnsum2.module import Module, register_module, hparam_registry
import nnsum2.layers


@register_module("fg.reconstructors.cloze")
class Cloze(Module):

    hparams = hparam_registry()

    @hparams(default=nnsum2.layers.Identity())
    def source_preprocessor(self):
        pass

    @hparams(default=nnsum2.layers.Identity())
    def target_preprocessor(self):
        pass

    @hparams()
    def attention_mechanism(self):
        pass

    @hparams()
    def postprocessor(self):
        pass

    def forward(self, source, cloze_indices, targets, embeddings, 
                targets_mask=None, return_attention=False):
        
        source_features = self.source_preprocessor(source)
        hidden_dim = source_features.size(2) // 2
        batch_size = source_features.size(0)
        seq_size = source_features.size(1)
        fwd_src, bwd_src = source_features \
            .view(batch_size, seq_size, 2, hidden_dim).split(1, dim=2)

        fwd_sel = (cloze_indices - 1).masked_fill(cloze_indices.eq(0), 0)
        fwd_sel = fwd_sel.unsqueeze(2).repeat(1, 1, hidden_dim)
        fwd_rep = fwd_src[:,:,0].gather(1, fwd_sel)

        bwd_sel = (cloze_indices + 1).masked_fill(cloze_indices.eq(0), 0)
        bwd_sel = bwd_sel.unsqueeze(2).repeat(1, 1, hidden_dim)
        bwd_rep = bwd_src[:,:,0].gather(1, bwd_sel)

        cloze_rep = torch.cat(
            [fwd_rep.permute(1,0,2), bwd_rep.permute(1,0,2)], dim=2)

        target_features, targets_mask = self.target_preprocessor(
            targets, inputs_mask=targets_mask)

        attn, _, attn_out = self.attention_mechanism(
            target_features, cloze_rep, key_mask=targets_mask)

        output = self.postprocessor(attn_out)

        n, bs, h = output.size()
       
        if output.numel() == 0:
            return None
        
        logits = torch.mm(output.view(-1, h), embeddings.t()).view(n, bs, -1)

        if return_attention:
            return logits, attn
        return logits

    def initialize_parameters(self):
        self.source_preprocessor.initialize_parameters()
        self.target_preprocessor.initialize_parameters()
        self.attention_mechanism.initialize_parameters()
        self.postprocessor.initialize_parameters()

    def set_dropout(self, dropout):
        self.source_preprocessor.set_dropout(dropout)
        self.target_preprocessor.set_dropout(dropout)
        self.attention_mechanism.set_dropout(dropout)
        self.postprocessor.set_dropout(dropout)
