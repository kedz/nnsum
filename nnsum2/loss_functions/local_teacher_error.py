import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry
import torch


@register_module("loss_functions.local_teacher_error")
class LocalTeacherError(Module):

    hparams = hparam_registry()

    @hparams(default="source_labels")
    def target_field(self):
        pass

    @hparams()
    def teacher(self):
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

    def _greedy_forward(self, search_state, batch):
        labels = batch[self.target_field]
        outputs = search_state.get_result("output")
        losses = self.teacher.local_errors(outputs.t(), labels)
        olp = search_state.get_result("output_log_probability")
        #print(olp.size())
        #print(olp)
        weighted_total_lp = (losses.t() * olp).sum(0)
        #print(weighted_total_lp)
        #print(batch["source_lengths"])

        lengths = outputs.ne(0).float().sum(0, keepdim=True)
        mean_weighted_lp = (
            weighted_total_lp / lengths
        ).mean()
        
        self._total_loss += mean_weighted_lp.item() * outputs.size(1)
        self._total_inputs += outputs.size(1)
        return mean_weighted_lp
        
    def _beam_forward(self, search_state, batch):
        outputs = search_state.get_result("output")
        time, batches, beams = outputs.size()

        labels = {field: label.view(-1, 1).repeat(1, beams).view(-1)
                  for field, label in batch[self.target_field].items()}
        losses = self.teacher.local_errors(
            outputs.view(time, batches * beams).contiguous().t(), labels)
        losses = losses.view(batches, beams, time).permute(2, 0, 1)
        olp = search_state.get_result("output_log_probability")
        weighted_total_lp = (losses * olp).sum(0)
        lengths = outputs.ne(0).float().sum(0, keepdim=True)

        mean_weighted_lp = (
            weighted_total_lp / lengths
        ).mean()

        self._total_loss += mean_weighted_lp.item() * batches * beams
        self._total_inputs += batches * beams
        return mean_weighted_lp

    def forward(self, search_state, batch):
        outputs = search_state.get_result("output")
        
        if outputs.dim() == 2:
            return self._greedy_forward(search_state, batch)
        else:
            return self._beam_forward(search_state, batch)
