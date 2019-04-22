import torch
from nnsum.seq2seq import RNNState


def cat(inputs, *args, **kwargs):
    if isinstance(inputs[0], RNNState):
        return RNNState.new_state(
            [torch.cat([item[i] for item in inputs], *args, **kwargs) 
             for i in range(len(inputs[0]))])
    else:
        return torch.cat(inputs, *args, **kwargs)

def pad_and_cat(inputs, *args, pad_dim=-1, pad_value=0, **kwargs):
    # make this work for rnn states someday.
    max_size = max([x.size(pad_dim) for x in inputs])
    padded_inputs = []
    for x in inputs:
        diff = max_size - x.size(pad_dim) 
        if diff > 0:
           sizes = list(x.size())
           sizes[pad_dim] = diff
           pad = x.new(*sizes).fill_(pad_value)
           padded_inputs.append(cat([x, pad], dim=pad_dim))

        else:
            padded_inputs.append(x)

    return cat(padded_inputs, *args, **kwargs)

def allclose(x, y, *args, **kwargs):
    if isinstance(x, RNNState):
        return all([torch.allclose(x_i, y_i, *args, **kwargs)
                    for x_i, y_i in zip(x, y)])
    else:
        return torch.allclose(x, y, *args, **kwargs)

def stack(inputs, *args, **kwargs):
    if isinstance(inputs[0], RNNState):
        return RNNState.new_state(
            [torch.stack([item[i] for item in inputs], *args, **kwargs) 
             for i in range(len(inputs[0]))])
    else:
        return torch.stack(inputs, *args, **kwargs)

def pad_and_stack(inputs, *args, pad_dim=-1, pad_value=0, **kwargs):
    # make this work for rnn states someday.
    max_size = max([x.size(pad_dim) if x.dim() > 0 else 0 for x in inputs])
    padded_inputs = []
    for x in inputs:
        diff = max_size - x.size(pad_dim) 
        if diff > 0:
           sizes = list(x.size())
           sizes[pad_dim] = diff
           pad = x.new(*sizes).fill_(pad_value)
           padded_inputs.append(cat([x, pad], dim=pad_dim))

        else:
            padded_inputs.append(x)

    return stack(padded_inputs, *args, **kwargs)


