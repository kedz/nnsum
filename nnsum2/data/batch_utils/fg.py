import torch
import random
import nnsum2.torch as ntorch


def word_prediction_batch(sources, targets, source_vocab, source_field, 
                          target_vocab, target_field,
                          context_size=10):
    source_indices = []
    target_indices = []
    prediction_targets = []

    for source, target in zip(sources, targets):
        
        redactable_indices = [
            index
            for index, redactable in enumerate(source['question_terms'])
            if redactable
        ]
        if len(redactable_indices) > 0:
            random.shuffle(redactable_indices)
            redact_position = redactable_indices[0]
        else:
            redactable_indices = list(range(len(source['question_terms'])))
            random.shuffle(redactable_indices)
            redact_position = redactable_indices[0]
        source_tokens = source["sequence"][source_field]
        left_context = source_tokens[
            max(0, redact_position - context_size):redact_position]
        right_context = source_tokens[
            redact_position + 1:redact_position + context_size + 1]
        pred_target = source_tokens[redact_position]
        if len(left_context) < context_size:
            diff = context_size - len(left_context)
            left_context = [source_vocab.pad_token] * diff + left_context
        if len(right_context) < context_size:
            diff = context_size - len(right_context)
            right_context = right_context + [source_vocab.pad_token] * diff 
        source_indices.append([
            source_vocab[x] 
            for x in left_context + [source_vocab.start_token] + right_context
        ])
        target_indices.append([
            target_vocab[token]
            for token in target["sequence"][target_field]
        ])
        prediction_targets.append(source_vocab[pred_target])

    prediction_targets = torch.LongTensor(prediction_targets)
    source_indices = torch.LongTensor(source_indices)
    max_tgt_len = max([len(x) for x in target_indices])
    for tgt in target_indices:
        diff = max_tgt_len - len(tgt)
        if diff > 0:
            tgt.extend([target_vocab.pad_index] * diff)
    target_indices = torch.LongTensor(target_indices)
    target_mask = target_indices.eq(target_vocab.pad_index)
    return {"source_features": {source_field: source_indices},
            "target_features": {target_field: target_indices},
            "target_mask": target_mask,
            "prediction_targets": prediction_targets}


def cloze_batch(sources, vocab, field):

    cloze_indices = []
    cloze_targets = []
    cloze_lengths = []
    for source in sources:
        # redactable indices start at 1 because we never redact the start token
        redactable_indices = [
            index
            for index, redactable in enumerate(source['question_terms'], 1)
            if redactable
        ]
        tokens = [vocab.start_token] + source["sequence"][field]
        target_tokens = [tokens[idx] for idx in redactable_indices]
        cloze_lengths.append(len(target_tokens))
        cloze_targets.append(
            torch.LongTensor([vocab[tok] for tok in target_tokens]))

        cloze_indices.append(torch.LongTensor(redactable_indices))
    cloze_indices = ntorch.pad_and_stack(cloze_indices)
    cloze_targets = ntorch.pad_and_stack(cloze_targets)
    cloze_lengths = torch.LongTensor(cloze_lengths)
    assert cloze_targets.size() == cloze_indices.size()

    return {"cloze_indices": cloze_indices, 
            "cloze_targets": cloze_targets,
            "cloze_lengths": cloze_lengths}

def cloze_batch2(sources, vocab, source_field, cloze_field):

    cloze_indices = []
    cloze_targets = []
    cloze_lengths = []
    for item in sources:
        # redactable indices start at 1 because we never redact the start token
        redactable_indices = [
            idx for idx, redact in enumerate(item["sequence"][cloze_field], 1)
            if redact
        ]
        tokens = [vocab.start_token] + item["sequence"][source_field]
        target_tokens = [tokens[idx] for idx in redactable_indices]
        cloze_lengths.append(len(target_tokens))
        cloze_targets.append(
            torch.LongTensor([vocab[tok] for tok in target_tokens]))

        cloze_indices.append(torch.LongTensor(redactable_indices))
    cloze_indices = ntorch.pad_and_stack(cloze_indices)
    cloze_targets = ntorch.pad_and_stack(cloze_targets)
    cloze_lengths = torch.LongTensor(cloze_lengths)
    assert cloze_targets.size() == cloze_indices.size()

    return {"cloze_indices": cloze_indices, 
            "cloze_targets": cloze_targets,
            "cloze_lengths": cloze_lengths}





def cloze_batch_to_gpu(batch, device):
    batch["cloze_indices"] = batch["cloze_indices"].cuda(device)
    return batch

def to_gpu(batch, device):
    batch["source_features"] = {
        field: values.cuda(device)
        for field, values in batch["source_features"].items()
    }                        
    batch["target_features"] = {
        field: values.cuda(device)
        for field, values in batch["target_features"].items()
    }                        
    batch["target_mask"] = batch["target_mask"].cuda(device)
    batch["prediction_targets"] = batch["prediction_targets"].cuda(device)
    return batch
