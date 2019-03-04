import torch
from nnsum.embedding_context import Vocab
from nnsum.util import batch_pad_and_stack_vector

from collections import OrderedDict


def batch_source(source_texts, source_vocab):
    
    if not isinstance(source_texts, (list, tuple)):
        raise ValueError()
        
    if isinstance(source_texts[0], (list, tuple)) \
            and isinstance(source_texts[0][0], str):
        return batch_source_from_list_of_list_of_strings(source_texts,
                                                         source_vocab)
    elif isinstance(source_texts[0], str): 
        return batch_source_from_list_of_strings(source_texts,
                                                 source_vocab)
    elif isinstance(source_texts[0], dict) \
            and isinstance(list(source_texts[0].values())[0], list):
        return batch_source_from_list_of_dict_of_list_of_strings(source_texts,
                                                                 source_vocab)
    elif isinstance(source_texts[0], dict) \
            and isinstance(list(source_texts[0].values())[0], str):
        return batch_source_from_list_of_dict_of_strings(source_texts,
                                                         source_vocab)
    else:
        raise Exception()

def batch_source_from_list_of_list_of_strings(source_texts, source_vocab):
    
    lengths = []
    data = []
    for tokens in source_texts:
        row = torch.LongTensor(
            [source_vocab[t] for t in tokens])
        lengths.append(row.size(0))
        data.append(row)
    data = batch_pad_and_stack_vector(data, source_vocab.pad_index)
    mask = torch.ByteTensor(data.size()).zero_()
    for i, length in enumerate(lengths):
        mask[i,length:].fill_(1)
    lengths = torch.LongTensor(lengths)
    return {"source_input_features": {"tokens": data},
            "source_lengths": lengths, "source_mask": mask}

def batch_source_from_list_of_strings(source_texts, source_vocab):
     
    lengths = []
    data = []
    for txt in source_texts:
        row = torch.LongTensor(
            [source_vocab[t] for t in txt.split()])
        lengths.append(row.size(0))
        data.append(row)
    data = batch_pad_and_stack_vector(data, source_vocab.pad_index)
    mask = torch.ByteTensor(data.size()).zero_()
    for i, length in enumerate(lengths):
        mask[i,length:].fill_(1)
    lengths = torch.LongTensor(lengths)
    return {"source_input_features": {"tokens": data},
            "source_lengths": lengths, "source_mask": mask}

def batch_source_from_list_of_dict_of_list_of_strings(source_dicts,
                                                      source_vocab_dicts):

    all_lengths = []
    all_data = {name: list() for name in source_vocab_dicts.keys()}
    
    for item in source_dicts:
        length = None
        for name in all_data.keys():
            tokens = item[name]
            vcb = source_vocab_dicts[name]
            row = torch.LongTensor([vcb[t] for t in tokens])
            all_data[name].append(row)
            if length is None:
                length = row.size(0)
            elif length != row.size(0):
                raise ValueError("All features must have the same length.")
        all_lengths.append(length)

    for name in all_data.keys():
        all_data[name] = batch_pad_and_stack_vector(
            all_data[name], source_vocab_dicts[name].pad_index)
    all_lengths = torch.LongTensor(all_lengths)

    for name, data in all_data.items():
        mask = data.eq(source_vocab_dicts[name].pad_index)
        break

    return {"source_input_features": all_data,
            "source_lengths": all_lengths,
            "source_mask": mask}

def batch_source_from_list_of_dict_of_strings(source_dicts,
                                              source_vocab_dicts):

    all_lengths = []
    all_data = {name: list() for name in source_vocab_dicts.keys()}
    
    for item in source_dicts:
        length = None
        for name in all_data.keys():
            text = item[name]
            vcb = source_vocab_dicts[name]
            row = torch.LongTensor([vcb[t] for t in text.split()])
            all_data[name].append(row)
            if length is None:
                length = row.size(0)
            elif length != row.size(0):
                raise ValueError("All features must have the same length.")
        all_lengths.append(length)

    for name in all_data.keys():
        all_data[name] = batch_pad_and_stack_vector(
            all_data[name], source_vocab_dicts[name].pad_index)
    all_lengths = torch.LongTensor(all_lengths)

    for name, data in all_data.items():
        mask = data.eq(source_vocab_dicts[name].pad_index)
        break

    return {"source_input_features": all_data,
            "source_lengths": all_lengths,
            "source_mask": mask}
