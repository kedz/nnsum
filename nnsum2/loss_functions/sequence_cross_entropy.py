import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.sequence_cross_entropy")
class SequenceCrossEntropy(Module):

    hparams = hparam_registry()

    @hparams(default=True)
    def use_logits(self):
        pass

    @hparams(default="target_logits")
    def logits_field(self):
        pass

    @hparams(default="target_log_probability")
    def log_probs_field(self):
        pass

    @hparams(default="target_output_features")
    def target_field(self):
        pass

    @hparams(default="target_lengths")
    def target_length_field(self):
        pass

    @hparams()
    def target_vocab_name(self):
        pass

    @hparams()
    def vocab(self):
        pass

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
        if self.use_logits:
            return self._logits_forward(forward_state, batch)
        else:
            return self._log_probs_forward(forward_state, batch)

    def _logits_forward(self, forward_state, batch):
        
        target_logits = forward_state[self.logits_field]
        targets = batch[self.target_field][self.target_vocab_name].t()

        assert len(self.vocab) == target_logits.size(2)
        target_logits_flat = target_logits.view(-1, len(self.vocab))
        targets_flat = targets.contiguous().view(-1)

        avg_xent = F.cross_entropy(target_logits_flat, targets_flat,
                                   ignore_index=self.vocab.pad_index) 
        num_el = batch[self.target_length_field].sum().item()
        
        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent

    def _log_probs_forward(self, forward_state, batch):

        target_log_probs = forward_state[self.log_probs_field]
        targets = batch[self.target_field][self.target_vocab_name].t()

        assert len(self.vocab) == target_log_probs.size(2)
        target_log_probs_flat = target_log_probs.view(-1, len(self.vocab))
        targets_flat = targets.contiguous().view(-1)

        avg_xent = F.nll_loss(target_log_probs, targets_flat,
                              ignore_index=self.vocab.pad_index) 
        num_el = batch[self.target_length_field].sum().item()
        
        self._total_loss += (avg_xent.item() * num_el)
        self._total_inputs += num_el
        return avg_xent
