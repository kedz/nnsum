import torch.nn.functional as F

def sequence_cross_entropy(logits, targets, pad_index=None):
    
    if isinstance(logits, list):
        tot_xent = 0.
        for step, logits_step in enumerate(logits):
            xent_step = F.cross_entropy(
                logits_step, targets[step], ignore_index=pad_index,
                reduction='none')
            tot_xent = tot_xent + xent_step.sum()
        return tot_xent 
    else:
        tot_xent = F.cross_entropy(logits.permute(1, 2, 0), targets.t(),
                                   ignore_index=pad_index, reduction="sum")
        return tot_xent
