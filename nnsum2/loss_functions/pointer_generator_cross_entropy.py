import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.pointer_generator_cross_entropy")
class PointerGeneratorCrossEntropy(Module):

    hparams = hparam_registry()
#
#    @hparams(default=True)
#    def use_logits(self):
#        pass

    @hparams(default="copy_targets")
    def copy_targets_field(self):
        pass

    @hparams(default="target_output_features")
    def target_field(self):
        pass

    @hparams(default="target_mask")
    def target_mask_field(self):
        pass

    @hparams(default="pointer_probability")
    def pointer_probability_field(self):
        pass

    @hparams(default="generator_probability")
    def generator_probability_field(self):
        pass


#    @hparams(default="target_logits")
#    def logits_field(self):
#        pass
#
#    @hparams(default="target_log_probability")
#    def log_probs_field(self):
#        pass
#
#
#    @hparams(default="target_lengths")
#    def target_length_field(self):
#        pass
#
    @hparams()
    def target_vocab_name(self):
        pass
#
#    @hparams()
#    def vocab(self):
#        pass

    def init_network(self):
        self.reset()
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def forward(self, forward_state, batch):
        extended_targets = batch[self.copy_targets_field]
        generator_targets = batch[self.target_field][self.target_vocab_name]

        # *_select tensors are steps x batch x 1 for gathering from their
        # respective probability distributions.
        ptr_select = extended_targets.t().unsqueeze(2)
        gen_select = generator_targets.t().unsqueeze(2)

        # If the ptr target does not match the gen target, then we should
        # only use the pointer distribution (i.e. mask the gen distribution)
        no_gen_mask = gen_select != ptr_select
 
        ptr_dist = forward_state[self.pointer_probability_field]
        ptr_prob = ptr_dist.gather(2, ptr_select)

        gen_dist = forward_state[self.generator_probability_field] 
        gen_prob = gen_dist.gather(2, gen_select).masked_fill(no_gen_mask, 0.)

        log_probs = torch.log(gen_prob + ptr_prob + 1e-8)

        target_mask = batch.get(self.target_mask_field, None)
        if target_mask is not None:
            log_probs = log_probs.masked_fill(target_mask.t().unsqueeze(2), 0)

        num_els = batch["target_lengths"].sum().item()
        total_nll = -log_probs.sum()
        self._total_loss += total_nll.item()
        self._total_inputs += num_els

        return total_nll / num_els
