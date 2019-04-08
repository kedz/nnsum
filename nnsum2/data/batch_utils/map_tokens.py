import torch


def map_tokens(inputs, field, vocab, start_token=False, stop_token=False,
               max_length=None):

    batch_size = len(inputs)
    max_steps = max([len(inp[field]) for inp in inputs])
    if max_length is not None:
        max_steps = min(max_steps, max_length)
    max_steps += start_token + stop_token
    indices = torch.LongTensor(batch_size, max_steps).fill_(vocab.pad_index)
        
    for batch, inp in enumerate(inputs):
        start_step = 0
        if start_token:
            indices[batch, start_step] = vocab.start_index
            start_step += 1
        inp_tokens = inp[field]
        if max_length is not None:
            inp_tokens = inp_tokens[:max_length]
        for step, token in enumerate(inp_tokens, start_step):
            indices[batch, step] = vocab[token]
        if stop_token:
            indices[batch, step + 1] = vocab.stop_index

    return indices
