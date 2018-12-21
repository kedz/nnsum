import torch


def binary_entropy(inputs, reduction="mean"):
    p = inputs
    not_p = 1 - inputs
    negative_entropy = p * torch.log(p) + not_p * torch.log(not_p)

    if reduction == "mean":
        return -negative_entropy.mean()
    elif reduction == "sum":
        return -negative_entropy.sum()
    elif reduction == "none":
        return -negative_entropy
    else:
        raise Exception("reduction must be 'mean', 'sum' or 'none'")
