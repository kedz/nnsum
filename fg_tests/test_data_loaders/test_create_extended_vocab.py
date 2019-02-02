import pytest
import torch
from nnsum.embedding_context import Vocab
from nnsum.data import seq2seq_batcher


@pytest.fixture(scope="module")
def sources():
    return [["A", "B", "C"],
            ["1", "B", "2", "C"],
            ["A", "3"]]

@pytest.fixture(scope="module")
def expected_dense_source_vocab_map():
    return torch.FloatTensor(
        [[[0,0,0,0, 0,0,0, 0,0,0],
          [0,0,0,0, 1,0,0, 0,0,0],
          [0,0,0,0, 0,1,0, 0,0,0],
          [0,0,0,0, 0,0,1, 0,0,0],
          [0,0,0,0, 0,0,0, 0,0,0],],
         [[0,0,0,0, 0,0,0, 0,0,0],
          [0,0,0,0, 0,0,0, 1,0,0],
          [0,0,0,0, 0,1,0, 0,0,0],
          [0,0,0,0, 0,0,0, 0,1,0],
          [0,0,0,0, 0,0,1, 0,0,0],],
         [[0,0,0,0, 0,0,0, 0,0,0],
          [0,0,0,0, 1,0,0, 0,0,0],
          [0,0,0,0, 0,0,0, 0,0,1],
          [0,0,0,0, 0,0,0, 0,0,0],
          [0,0,0,0, 0,0,0, 0,0,0],]])

@pytest.fixture(scope="module")
def targets():
    return [["B", "C", "<eos>"],
            ["1", "2", "C", "<eos>"],
            ["?", "3", "<eos>"]]

@pytest.fixture(scope="module")
def expected_copy_targets():
    return torch.LongTensor(
        [[5, 6, 3, 0],
         [7, 8, 6, 3],
         [1, 9, 3, 0]])

@pytest.fixture(scope="module")
def target_vocab():
    return Vocab.from_word_list(["A", "B", "C",], start="<sos>", stop="<eos>",
                                pad="<pad>", unk="<unk>")

@pytest.fixture(scope="module")
def expected_ext_vocab():
    return Vocab.from_word_list(["1", "2", "3"])

def test_extended_vocab(expected_ext_vocab, sources, target_vocab):
    ext_vocab = seq2seq_batcher._batch_create_extended_vocab_impl(
        sources, target_vocab)

    for exp, act in zip(expected_ext_vocab.enumerate(), ext_vocab.enumerate()):
        assert exp == act

@pytest.fixture(scope="module")
def dense_source_vocab_map(sources, target_vocab, expected_ext_vocab):
    return seq2seq_batcher._create_dense_vocab_map(
        sources, expected_ext_vocab, target_vocab)

def test_dense_source_vocab_map(expected_dense_source_vocab_map,
                                dense_source_vocab_map):
    assert torch.all(expected_dense_source_vocab_map == dense_source_vocab_map)

@pytest.fixture(scope="module")
def copy_targets(targets, target_vocab, expected_ext_vocab):
    return seq2seq_batcher._batch_create_copy_targets(
        targets, target_vocab, expected_ext_vocab)

def test_copy_targets(expected_copy_targets, copy_targets):
    assert torch.all(expected_copy_targets == copy_targets)
