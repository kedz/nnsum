import torch
import torch.nn as nn


def target_getter(batch):
    return (batch["copy_targets"],
            batch["target_output_features"]["tokens"], 
            batch["target_lengths"])

class PointerGeneratorCrossEntropyLoss(nn.Module):
    def __init__(self, pointer_probability_field="pointer_probability",
                 generator_probability_field="generator_probability",
                 extended_target_field="copy_targets",
                 generator_target_field="target_output_features",
                 target_mask_field="target_mask"):
    
        super(PointerGeneratorCrossEntropyLoss, self).__init__()
        self._pointer_probability_field = pointer_probability_field
        self._generator_probability_field = generator_probability_field
        self._extended_target_field = extended_target_field
        self._generator_target_field = generator_target_field
        self._target_mask_field = target_mask_field
        self._total_loss = 0
        self._total_inputs = 0

    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    @property
    def pointer_probability_field(self):
        return self._pointer_probability_field

    @property
    def generator_probability_field(self):
        return self._generator_probability_field

    @property
    def extended_target_field(self):
        return self._extended_target_field

    @property
    def generator_target_field(self):
        return self._generator_target_field

    @property
    def target_mask_field(self):
        return self._target_mask_field
    
    def forward(self, forward_state, batch):

        extended_targets = batch[self.extended_target_field]
        
        generator_targets = batch[self.generator_target_field]
        assert len(generator_targets) == 1
        generator_targets = list(generator_targets.values())[0]

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

        log_probs = torch.log(gen_prob + ptr_prob)

        target_mask = batch.get(self.target_mask_field, None)
        if target_mask is not None:
            log_probs = log_probs.masked_fill(target_mask.t().unsqueeze(2), 0)
            num_els = log_probs.numel() - target_mask.long().sum().item()
        else:
            num_els = log_probs.numel()
        total_nll = -log_probs.sum()
        self._total_loss += total_nll.item()
        self._total_inputs += num_els

        return total_nll / num_els
