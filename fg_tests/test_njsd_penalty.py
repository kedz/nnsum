import torch
from nnsum.loss import binary_entropy

def njsd_penalty(probs):
    prob_masks = [p.gt(.5) for p in probs]
    active_counts = sum([m.float() for m in prob_masks])
    inactive = active_counts.eq(0)
    weights = 1 / active_counts.masked_fill(active_counts.eq(0), 1.)

    sum_probs = weights * sum([p.masked_fill(~m, 0) 
                               for p, m in zip(probs, prob_masks)])
    jnt_entropy = binary_entropy(
        sum_probs, reduction="none").masked_fill(inactive, 0)

    ind_entropies = []
    for p, m in zip(probs, prob_masks):
        ind_entropy = binary_entropy(p, reduction="none").masked_fill(~m, 0)
        ind_entropies.append(ind_entropy)
    ind_entropies = weights * sum(ind_entropies)

    jsd = jnt_entropy - ind_entropies

    counts = jsd.ne(0).float().sum()
    njsd = jsd.sum() * -1.
    return njsd, counts

p1 = torch.sigmoid(torch.FloatTensor(3,8).normal_())
p2 = torch.sigmoid(torch.FloatTensor(3,8).normal_())
njsd_penalty([p1, p2])

