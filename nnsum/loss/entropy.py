import torch


def entropy(logits, reduction="mean"):
    p = torch.softmax(logits, dim=1)
    lp = torch.log_softmax(logits, dim=1)
    negative_entropy = (p * lp).sum(dim=1)

    if reduction == "mean":
        return -negative_entropy.mean()
    elif reduction == "sum":
        return -negative_entropy.sum()
    elif reduction == "none":
        return -negative_entropy
    else:
        raise Exception("reduction must be 'mean', 'sum' or 'none'")
