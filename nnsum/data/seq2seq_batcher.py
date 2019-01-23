import torch
from nnsum.embedding_context import Vocab
from nnsum.util import batch_pad_and_stack_vector


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

def batch_target(target_texts, target_vocab):
    
    if not isinstance(target_texts, (list, tuple)):
        raise ValueError()
        
    if isinstance(target_texts[0], (list, tuple)) \
            and isinstance(target_texts[0][0], str):
        return batch_target_from_list_of_list_of_strings(target_texts,
                                                         target_vocab)
    elif isinstance(target_texts[0], str): 
        return batch_target_from_list_of_strings(target_texts,
                                                 target_vocab)
    elif isinstance(target_texts[0], dict) \
            and isinstance(list(target_texts[0].values())[0], list):
        return batch_target_from_list_of_dict_of_list_of_strings(target_texts,
                                                                 target_vocab)
    elif isinstance(target_texts[0], dict) \
            and isinstance(list(target_texts[0].values())[0], str):
        return batch_target_from_list_of_dict_of_strings(target_texts,
                                                         target_vocab)
    else:
        raise Exception()

def batch_copy_alignments(source, target, target_vocab, alignments,
                          mixture_copy_prob=.5):

    if not isinstance(source, (list, tuple)):
        raise ValueError()

    if not isinstance(target, (list, tuple)):
        raise ValueError()

    if isinstance(target[0], (list, tuple)) \
            and isinstance(target[0][0], str):
        return _batch_cpy_align_from_list_of_list_of_strs(
            source, target, target_vocab, alignments, 
            mixture_copy_prob=mixture_copy_prob)

    elif isinstance(target[0], str): 
        return _batch_cpy_align_from_list_of_strs(
            source, target, target_vocab, alignments, 
            mixture_copy_prob=mixture_copy_prob)

    elif isinstance(target[0], dict) \
            and isinstance(list(target[0].values())[0], list):
        return _batch_cpy_align_from_list_of_dict_of_list_of_strs(
            source, target, target_vocab, alignments, 
            mixture_copy_prob=mixture_copy_prob)

    elif isinstance(target[0], dict) \
            and isinstance(list(target[0].values())[0], str):
        return _batch_cpy_align_from_list_of_dict_of_strs(
            source, target, target_vocab, alignments, 
            mixture_copy_prob=mixture_copy_prob)

    raise Exception()


def batch_source_from_list_of_list_of_strings(source_texts, source_vocab):
    
    lengths = []
    data = []
    for tokens in source_texts:
        row = torch.LongTensor(
            [source_vocab.start_index] + [source_vocab[t] for t in tokens])
        lengths.append(row.size(0))
        data.append(row)
    lengths = torch.LongTensor(lengths)
    data = batch_pad_and_stack_vector(data, source_vocab.pad_index)
    return {"source_input_features": {"tokens": data},
            "source_lengths": lengths}

def batch_target_from_list_of_list_of_strings(target_texts, target_vocab):
    
    lengths = []
    data_in = []
    data_out = []
    for tokens in target_texts:
        idxs = [target_vocab[t] for t in tokens]
        row_in = torch.LongTensor([target_vocab.start_index] + idxs)
        row_out = torch.LongTensor(idxs + [target_vocab.stop_index]) 
        lengths.append(row_in.size(0))
        data_in.append(row_in)
        data_out.append(row_out)
    lengths = torch.LongTensor(lengths)
    data_in = batch_pad_and_stack_vector(data_in, target_vocab.pad_index)
    data_out = batch_pad_and_stack_vector(data_out, target_vocab.pad_index)
    return {"target_input_features": {"tokens": data_in},
            "target_output_features": {"tokens": data_out},
            "target_lengths": lengths}

def _batch_cpy_align_from_list_of_list_of_strs(srcs, tgts, vcb, aligns,
                                               mixture_copy_prob=.5):
   
    src_vocab = Vocab.from_word_list([w for src in srcs for w in src])
    source_vocab_map = torch.FloatTensor(
        len(srcs), 1 + max([len(src) for src in srcs]), len(src_vocab)).zero_()

    for i, src in enumerate(srcs):
        for j, tok in enumerate(src, 1):
            source_vocab_map[i,j, src_vocab[tok]] = 1

    copy_probs = []
    copy_targets = []
    for src, tgt, align in zip(srcs, tgts, aligns):
        cpy_tgts = []
        cpy_prbs = []
        for tok, idx in zip(tgt, align):
            if idx > -1:
                cpy_tgts.append(src_vocab[src[idx]])
                if vcb[tok] == vcb.unknown_index:
                    cpy_prbs.append(1.)
                else:
                    cpy_prbs.append(mixture_copy_prob)
            else:
                cpy_tgts.append(-1)
                cpy_prbs.append(0.)

        cpy_tgts.append(-1)
        cpy_prbs.append(0.)
        copy_targets.append(torch.LongTensor(cpy_tgts))
        copy_probs.append(torch.FloatTensor(cpy_prbs))
    copy_targets = batch_pad_and_stack_vector(copy_targets, -1)
    copy_probs = batch_pad_and_stack_vector(copy_probs, -1)

    return {"copy_probabilty": copy_probs,
            "copy_targets":  copy_targets,
            "source_vocab_map": source_vocab_map}

def batch_source_from_list_of_strings(source_texts, source_vocab):
     
    lengths = []
    data = []
    for txt in source_texts:
        row = torch.LongTensor(
            [source_vocab.start_index] + [source_vocab[t] 
                                          for t in txt.split()])
        lengths.append(row.size(0))
        data.append(row)
    lengths = torch.LongTensor(lengths)
    data = batch_pad_and_stack_vector(data, source_vocab.pad_index)
    return {"source_input_features": {"tokens": data},
            "source_lengths": lengths}

def batch_target_from_list_of_strings(target_texts, target_vocab):
    
    lengths = []
    data_in = []
    data_out = []
    for texts in target_texts:
        idxs = [target_vocab[t] for t in texts.split()]
        row_in = torch.LongTensor([target_vocab.start_index] + idxs)
        row_out = torch.LongTensor(idxs + [target_vocab.stop_index]) 
        lengths.append(row_in.size(0))
        data_in.append(row_in)
        data_out.append(row_out)
    lengths = torch.LongTensor(lengths)
    data_in = batch_pad_and_stack_vector(data_in, target_vocab.pad_index)
    data_out = batch_pad_and_stack_vector(data_out, target_vocab.pad_index)
    return {"target_input_features": {"tokens": data_in},
            "target_output_features": {"tokens": data_out},
            "target_lengths": lengths}

def _batch_cpy_align_from_list_of_strs(srcs, tgts, vcb, aligns,
                                       mixture_copy_prob=.5):
   
    srcs = [src.split() for src in srcs]
    tgts = [tgt.split() for tgt in tgts]
    return _batch_cpy_align_from_list_of_list_of_strs(
        srcs, tgts, vcb, aligns, mixture_copy_prob=mixture_copy_prob)

def batch_source_from_list_of_dict_of_list_of_strings(source_dicts,
                                                      source_vocab_dicts):

    all_lengths = []
    all_data = {name: list() for name in source_dicts[0].keys()}
    
    for item in source_dicts:
        length = None
        for name, tokens in item.items():
            vcb = source_vocab_dicts[name]
            row = torch.LongTensor(
                [vcb.start_index] + [vcb[t] for t in tokens])
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

    return {"source_input_features": all_data,
            "source_lengths": all_lengths}

def batch_target_from_list_of_dict_of_list_of_strings(target_dicts,
                                                      target_vocab_dicts):

    all_lengths = []
    all_data_in = {name: list() for name in target_dicts[0].keys()}
    all_data_out = {name: list() for name in target_dicts[0].keys()}
    
    for item in target_dicts:
        length = None
        for name, tokens in item.items():
            vcb = target_vocab_dicts[name]
            idxs = [vcb[t] for t in tokens]
            row_in = torch.LongTensor([vcb.start_index] + idxs)
            row_out = torch.LongTensor(idxs + [vcb.stop_index])
            all_data_in[name].append(row_in)
            all_data_out[name].append(row_out)
            if length is None:
                length = row_in.size(0)
            elif length != row_in.size(0):
                raise ValueError("All features must have the same length.")
        all_lengths.append(length)

    for name in all_data_in.keys():
        all_data_in[name] = batch_pad_and_stack_vector(
            all_data_in[name], target_vocab_dicts[name].pad_index)
        all_data_out[name] = batch_pad_and_stack_vector(
            all_data_out[name], target_vocab_dicts[name].pad_index)
    all_lengths = torch.LongTensor(all_lengths)

    return {"target_input_features": all_data_in,
            "target_output_features": all_data_out,
            "target_lengths": all_lengths}

def _batch_cpy_align_from_list_of_dict_of_list_of_strs(srcs, tgts, vcb, aligns,
                                                       mixture_copy_prob=.5):

    assert len(tgts[0]) == 1
    name = list(tgts[0].keys())[0]
    srcs = [src[name] for src in srcs]
    tgts = [tgt[name] for tgt in tgts]
    if isinstance(vcb, dict):
        vcb = vcb[name]
    return _batch_cpy_align_from_list_of_list_of_strs(
        srcs, tgts, vcb, aligns, mixture_copy_prob=mixture_copy_prob)

def batch_source_from_list_of_dict_of_strings(source_dicts,
                                              source_vocab_dicts):

    all_lengths = []
    all_data = {name: list() for name in source_dicts[0].keys()}
    
    for item in source_dicts:
        length = None
        for name, text in item.items():
            vcb = source_vocab_dicts[name]
            row = torch.LongTensor(
                [vcb.start_index] + [vcb[t] for t in text.split()])
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

    return {"source_input_features": all_data,
            "source_lengths": all_lengths}

def batch_target_from_list_of_dict_of_strings(target_dicts,
                                              target_vocab_dicts):
    all_lengths = []
    all_data_in = {name: list() for name in target_dicts[0].keys()}
    all_data_out = {name: list() for name in target_dicts[0].keys()}
    
    for item in target_dicts:
        length = None
        for name, texts in item.items():
            vcb = target_vocab_dicts[name]
            idxs = [vcb[t] for t in texts.split()]
            row_in = torch.LongTensor([vcb.start_index] + idxs)
            row_out = torch.LongTensor(idxs + [vcb.stop_index])
            all_data_in[name].append(row_in)
            all_data_out[name].append(row_out)
            if length is None:
                length = row_in.size(0)
            elif length != row_in.size(0):
                raise ValueError("All features must have the same length.")
        all_lengths.append(length)

    for name in all_data_in.keys():
        all_data_in[name] = batch_pad_and_stack_vector(
            all_data_in[name], target_vocab_dicts[name].pad_index)
        all_data_out[name] = batch_pad_and_stack_vector(
            all_data_out[name], target_vocab_dicts[name].pad_index)
    all_lengths = torch.LongTensor(all_lengths)

    return {"target_input_features": all_data_in,
            "target_output_features": all_data_out,
            "target_lengths": all_lengths}

def _batch_cpy_align_from_list_of_dict_of_strs(srcs, tgts, vcb, aligns,
                                               mixture_copy_prob=.5):

    assert len(tgts[0]) == 1
    name = list(tgts[0].keys())[0]
    srcs = [src[name].split() for src in srcs]
    tgts = [tgt[name].split() for tgt in tgts]
    if isinstance(vcb, dict):
        vcb = vcb[name]
    return _batch_cpy_align_from_list_of_list_of_strs(
        srcs, tgts, vcb, aligns, mixture_copy_prob=mixture_copy_prob)
