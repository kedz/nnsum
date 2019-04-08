import pytest
import torch
from nnsum.embedding_context import Vocab
from nnsum2.data import batch_utils


@pytest.fixture(scope="module")
def sources():
    return [
        {"tokens": ["A", "B", "C"],},
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

    copy_targets = batch_utils.map_tokens(targets, "tokens", extended_vocab, 
                                          stop_token=True)

    assert torch.all(expected_copy_targets == copy_targets)
