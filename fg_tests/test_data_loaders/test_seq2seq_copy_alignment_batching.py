import pytest
import torch
import nnsum.data.seq2seq_batcher as seq2seq_batcher
from nnsum.embedding_context import Vocab


def batch_param_gen():
    return [{"mixture_copy_prob": .5, "sparse_map": False},
            {"mixture_copy_prob": .5, "sparse_map": True},
            {"mixture_copy_prob": .2, "sparse_map": True}]

def batch_param_names(params):
    return "mix_cpy_prob={mixture_copy_prob}:sparse={sparse_map}".format(
        **params)

@pytest.fixture(scope="module", 
                params=batch_param_gen(),
                ids=batch_param_names)
def batch_params(request):
    return request.param

@pytest.fixture(
    scope="module",
    params=["list_of_list_of_strings", "list_of_strings", 
            "list_of_dict_of_list_of_strings", "list_of_dict_of_strings"])
def batch(request, batch_params):
    if request.param == "list_of_list_of_strings":
        src = [["E1", "E2", "C", "D"],
               ["A", "B", "R1", "C"],
               ["B", "A", "7"]]
        tgt = [["1", "E2", "2", "3"],
               ["4", "5", "?"],
               ["6", "7", "8", "7"]]
    elif request.param == "list_of_strings":
        src = ["E1 E2 C D",
               "A B R1 C",
               "B A 7"]
        tgt = ["1 E2 2 3",
               "4 5 ?",
               "6 7 8 7"]
    elif request.param == "list_of_dict_of_list_of_strings":
        src = [
            {"tokens": ["E1", "E2", "C", "D"]},
            {"tokens": ["A", "B", "R1", "C"]},
            {"tokens": ["B", "A", "7"]}
        ]

        tgt = [
            {"tokens": ["1", "E2", "2", "3"]},
            {"tokens": ["4", "5", "?"]},
            {"tokens": ["6", "7", "8", "7"]}
        ]
    elif request.param == "list_of_dict_of_strings":
        src = [
            {"tokens": "E1 E2 C D"},
            {"tokens": "A B R1 C"},
            {"tokens": "B A 7"}
        ]

        tgt = [
            {"tokens": "1 E2 2 3"},
            {"tokens": "4 5 ?"},
            {"tokens": "6 7 8 7"}
        ]
    else:
        raise ValueError("Invalid parameter: {}".format(request.param))

    alignment = [[-1, 1, -1, -1],
                 [-1, -1, 2],
                 [-1,-1, -1, -1]]

    vocab = Vocab.from_word_list(
        ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
        pad="<PAD>", unk="<UNK>", start="<START>", stop="<STOP>")

    return seq2seq_batcher.batch_copy_alignments(
        src, tgt, vocab, alignment, **batch_params)

@pytest.fixture(scope="module")
def expected_copy_probability(batch_params):
    mcp = batch_params["mixture_copy_prob"]
    return torch.FloatTensor(
        [[0, mcp,  0, 0,  0],
         [0,   0,  1, 0, -1],
         [0,   0,  0, 0,  0]])

def test_copy_probability(batch, expected_copy_probability):
    assert torch.all(batch["copy_probability"] == expected_copy_probability)

@pytest.fixture(scope="module")
def expected_source_vocab_map(batch_params):
    idxs = torch.LongTensor(
        #E1  E2   C  D      A   B   R1  C    B  A  7 
        [[1,  2,  3, 4,     6,  7,  8,  9,   11,12,13],
         [13,12, 14, 15,   16, 17, 18, 14,   17,16,10]])
    vals = torch.FloatTensor([1.] * 11)
    src_vcb_map = torch.sparse.FloatTensor(idxs, vals, torch.Size([15,19]))

    if batch_params["sparse_map"]:
        return src_vcb_map.to_dense()
    else:
        return src_vcb_map.to_dense().view(3,5,19)

@pytest.fixture(scope="module")
def actual_source_vocab_map(batch_params, batch):
    if batch_params["sparse_map"]:
        return batch["source_vocab_map"].to_dense()
    else:
        return batch["source_vocab_map"]

def test_source_vocab_map(actual_source_vocab_map, expected_source_vocab_map):
    assert torch.all(actual_source_vocab_map == expected_source_vocab_map)

@pytest.fixture(scope="module")
def expected_copy_targets():
    return torch.LongTensor(
        [[4, 12, 5, 6, 3],
         [7,  8, 18, 3, 0],
         [9, 10, 11, 10, 3]])

def test_copy_targets(batch, expected_copy_targets):
    assert torch.all(batch["copy_targets"] == expected_copy_targets)

#  0 1 2 3  4    5    6     7    8    9    10   11   12
#          ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
#          13 14 15 16 17 18
#          E1 C  D  A  B  R1 
#        src = [["E1", "E2", "C", "D"],
#               ["A", "B", "R1", "C"],
#               ["B", "A", "7"]]
#        tgt = [["1", "E2", "2", "3"],
#               ["4", "5", "?"],
#               ["6", "7", "8", "7"]]
#
