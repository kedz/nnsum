import pytest
import torch
from nnsum.embedding_context import Vocab
from nnsum2.data import batch_utils
from nnsum2.seq2seq import RNNCopyDecoder


@pytest.fixture(scope="module")
def sources():
    return [
        {"tokens": ["A", "B", "A"],},
        {"tokens":["1", "B", "2", "C"],},
        {"tokens": ["A", "3"]},
    ]

@pytest.fixture(scope="module")
def targets():
    return [
        {"tokens": ["B", "C",],},
        {"tokens": ["1", "2", "C",],},
        {"tokens": ["?", "3",],},
    ]

@pytest.fixture(scope="module")
def target_vocab():
    return Vocab.from_word_list(["A", "B", "C",], start="<sos>", stop="<eos>",
                                pad="<pad>", unk="<unk>")

@pytest.fixture(scope="module")
def expected_copy_targets():
    return torch.LongTensor(
        [[5, 6, 3, 0],
         [7, 8, 6, 3],
         [1, 9, 3, 0]])

def test_copy_targets(sources, targets, target_vocab, expected_copy_targets):

    extended_vocab = batch_utils.s2s.extend_vocab(
        sources, "tokens", target_vocab)

    inputs = batch_utils.map_tokens(sources, "tokens", extended_vocab,
                                    start_token=True)
    inputs_mask = inputs.eq(extended_vocab.pad_index)

    copy_targets = batch_utils.map_tokens(targets, "tokens", extended_vocab, 
                                          stop_token=True)

    batch_size, target_steps = copy_targets.size()
    source_steps = inputs.size(1)

    attn_logits = torch.FloatTensor(
        target_steps, batch_size, source_steps).normal_()
    attn_logits = attn_logits.masked_fill(
        inputs_mask.unsqueeze(0), float("-inf"))

    attn = torch.nn.Parameter(torch.softmax(attn_logits, dim=2))

    expected_copy_probs = torch.FloatTensor(
        target_steps, batch_size, len(extended_vocab)).fill_(0)

    for target_step in range(target_steps):

        for batch, source in enumerate(sources):
            for source_step, token in enumerate(
                    [extended_vocab.start_token] + source["tokens"]):
                idx = extended_vocab[token]
                p = attn[target_step, batch, source_step].item()
                expected_copy_probs[target_step, batch, idx] += p

    copy_probs = RNNCopyDecoder._map_attention_to_vocab(
            None, attn, inputs, len(extended_vocab))

    assert torch.allclose(expected_copy_probs, copy_probs)

    copy_probs.sum().backward()

    assert torch.all(attn.grad.eq(1.))
