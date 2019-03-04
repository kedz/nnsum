import torch.nn as nn


class AttentionCoverage(nn.Module):
    def __init__(self, input_field="context_attention", 
                 source_mask_field="source_mask", 
                 target_mask_field="target_mask",
                 iterative=False):
        super(AttentionCoverage, self).__init__()
        self._input_field = input_field
        self._source_mask_field = source_mask_field
        self._target_mask_field = target_mask_field
        self._iterative = iterative
        self._time_dim = 0
        self._batch_dim = 1
        self._total_inputs = 0
        self._total_loss = 0

    def reset(self):
        self._total_inputs = 0
        self._total_loss = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    @property
    def input_field(self):
        return self._input_field

    @property
    def source_mask_field(self):
        return self._source_mask_field

    @property
    def target_mask_field(self):
        return self._target_mask_field

    @property
    def iterative(self):
        return self._iterative
    
    def forward(self, forward_state, batch):
        eps = 1e-7
        attention = forward_state[self.input_field] + eps
        src_mask = batch.get(self.source_mask_field, None)
        tgt_mask = batch.get(self.target_mask_field, None)
        
        if self.iterative:
            clamped_scores = attention.cumsum(self._time_dim).clamp(0,1)
            if src_mask is not None:
                clamped_scores = clamped_scores.masked_fill(
                    src_mask.unsqueeze(0), 1.)
            if tgt_mask is not None:
                clamped_scores = clamped_scores.masked_fill(
                    tgt_mask.t().unsqueeze(2), 1.)

        else:
            if tgt_mask is not None:
                attention = attention.masked_fill(tgt_mask.t().unsqueeze(2), 0)
            clamped_scores = attention.sum(self._time_dim).clamp(0, 1)
            if src_mask is not None:
                clamped_scores = clamped_scores.masked_fill(src_mask, 1.)
        #print(tgt_mask)
        #print(clamped_scores)
        loss = -clamped_scores.log().sum()

        self._total_inputs += attention.size(self._batch_dim)
        self._total_loss += loss.item()

        return loss
