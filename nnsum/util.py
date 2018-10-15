import torch


def batch_pad_and_stack_matrix(tensors, pad):
    
    assert len(set([t.dim() for t in tensors])) == 1

    sizes = torch.stack([torch.LongTensor([*t.size()]) for t in tensors])
    max_sizes, _ = sizes.max(0) 

    batch_size = len(tensors)
    batch_tensor = tensors[0].new(batch_size, *max_sizes).fill_(pad)

    for t, tsr in enumerate(tensors):
        tslice = batch_tensor[t,:tsr.size(0),:tsr.size(1)]
        tslice.copy_(tsr)
    return batch_tensor

def batch_pad_and_stack_vector(tensors, pad):

    max_size = max([t.size(0) for t in tensors])

    batch_size = len(tensors)
    batch_tensor = tensors[0].new(batch_size, max_size).fill_(pad)

    for t, tsr in enumerate(tensors):
        tslice = batch_tensor[t,:tsr.size(0)]
        tslice.copy_(tsr)
    return batch_tensor
