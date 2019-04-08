import torch
from collections import OrderedDict
from nnsum.embedding_context import Vocab
from .map_tokens import map_tokens


def source(batch, vocabs, sequence_field="sequence", max_length=None):
    source_data = []
    for field, vocab in vocabs.items():
        indices = map_tokens([item[sequence_field] for item in batch],
                             field, vocab, start_token=True,
                             max_length=max_length)
        lengths = []
        for item in batch:
            toks = item[sequence_field][field]
            if max_length is not None:
                toks = toks[:max_length]
            lengths.append(len(toks) + 1)

        lengths = torch.LongTensor(lengths)
        mask = indices.eq(vocab.pad_index)
        source_data.append((field, indices, lengths, mask))

    length_0 = source_data[0][2]
    for data in source_data[1:]:
        length_i = data[2]
        if not torch.all(length_i == length_0):
            raise Exception("Field {} has inconsistent length".format(data[0]))

    mask_0 = source_data[0][3]
    for data in source_data[1:]:
        mask_i = data[3]
        if not torch.all(mask_i == mask_0):
            raise Exception("Field {} has inconsistent mask".format(data[0]))

    source_input_features = OrderedDict()
    for datum in source_data:
        source_input_features[datum[0]] = datum[1]
    
    return {"source_input_features": source_input_features,
            "source_lengths": length_0,
            "source_mask": mask_0}
  
def target(batch, vocabs, sequence_field="sequence", max_length=None):
    target_data = []
    inputs = [item[sequence_field] for item in batch]
    for field, vocab in vocabs.items():

        in_feats, out_feats = map_input_output_tokens(inputs, field, vocab,
                                                      max_length=max_length)
        lengths = []
        for item in batch:
            toks = item[sequence_field][field]
            if max_length is not None:
                toks = toks[:max_length]
            lengths.append(len(toks))

        lengths = torch.LongTensor(lengths)
        mask = in_feats.eq(vocab.pad_index)
        target_data.append((field, in_feats, out_feats, lengths, mask))

    length_0 = target_data[0][3]
    for data in target_data[1:]:
        length_i = data[3]
        if not torch.all(length_i == length_0):
            raise Exception("Field {} has inconsistent length".format(data[0]))

    mask_0 = target_data[0][4]
    for data in target_data[1:]:
        mask_i = data[4]
        if not torch.all(mask_i == mask_0):
            raise Exception("Field {} has inconsistent mask".format(data[0]))

    target_input_features = OrderedDict()
    target_output_features = OrderedDict()
    for datum in target_data:
        target_input_features[datum[0]] = datum[1]
        target_output_features[datum[0]] = datum[2]
    
    output = {"target_input_features": target_input_features,
              "target_output_features": target_output_features,
              "target_lengths": length_0,
              "target_mask": mask_0}

    if "reference_string" in batch[0]:
        output["target_reference_strings"] = [[item["reference_string"]]
                                              for item in batch]
    
    return output

def extend_vocab(inputs, field, vocab, sequence_field="sequence", 
                 max_length=None):
    ext_words = OrderedDict()
    for inp in inputs:
        inp_tokens = inp[sequence_field][field]
        if max_length is not None:
            inp_tokens = inp_tokens[:max_length]
        for token in inp_tokens:
            if token not in vocab:
                ext_words[token] = True 

    ext_word_list = list(vocab)
    ext_word_list.extend(ext_words.keys())

    return Vocab.from_list(ext_word_list, 
                           pad=vocab.pad_token, 
                           unk=vocab.unknown_token, 
                           start=vocab.start_token,
                           stop=vocab.stop_token,
                           counts=vocab.counts)

def copy_sequence(batch, field, start_token, sequence_field="sequence",
                  max_length=None):
    data = []
    for item in batch:
        inp_tokens = item[sequence_field][field]
        if max_length is not None:
            inp_tokens = inp_tokens[:max_length]
        seq = [start_token] + inp_tokens
        data.append(seq)
    return data

def discrete_controls(batch, control_vocabs, control_field="controls"):
    controls = OrderedDict()
    for field, vocab in control_vocabs.items():
        values = torch.LongTensor([vocab[item[control_field][field]] 
                                   for item in batch])
        controls[field] = values
    return controls

def map_input_output_tokens(inputs, field, vocab, max_length=None):

    batch_size = len(inputs)
    max_steps = max([len(inp[field]) for inp in inputs]) 

    if max_length is not None:
        max_steps = min(max_steps, max_length)
    max_steps += 1
    in_feats = torch.LongTensor(batch_size, max_steps).fill_(vocab.pad_index)
    out_feats = torch.LongTensor(batch_size, max_steps).fill_(vocab.pad_index)
        
    in_feats[:, 0].fill_(vocab.start_index)
    for batch, inp in enumerate(inputs):
        inp_tokens = inp[field]
        if max_length is not None:
            inp_tokens = inp_tokens[:max_length]
        for step, token in enumerate(inp_tokens):
            idx = vocab[token]
            in_feats[batch, step + 1] = idx
            out_feats[batch, step] = idx
        out_feats[batch, step + 1] = vocab.stop_index

    return in_feats, out_feats
