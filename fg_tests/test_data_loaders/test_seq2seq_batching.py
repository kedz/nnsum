import pytest
import torch
import nnsum.data.seq2seq_batcher as seq2seq_batcher
from nnsum.embedding_context import Vocab


def test_batch_source_list_of_list_of_strings():

    vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                 pad="<PAD>", unk="<UNK>", start="<START>")
    data = [["A", "B", "C",],
            ["B", "D"],
            ["X", "Y", "!", "Z"]]

    expected = {"source_input_features": {
                    "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                [2, 4, 6, 0, 0],
                                                [2, 7, 8, 1, 9]])},
                "source_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_source(
        data, vocab)

    assert len(actual) == 2
    assert "source_input_features" in actual
    assert len(actual["source_input_features"]) == 1
    assert "tokens" in actual["source_input_features"]
    assert torch.all(
        actual["source_input_features"]["tokens"] 
            == expected["source_input_features"]["tokens"])
    assert "source_lengths" in actual
    assert torch.all(actual["source_lengths"] == expected["source_lengths"])

def test_batch_source_list_of_strings():

    vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                 pad="<PAD>", unk="<UNK>", start="<START>")
    data = ["A B C",
            "B D",
            "X Y ! Z"]

    expected = {"source_input_features": {
                    "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                [2, 4, 6, 0, 0],
                                                [2, 7, 8, 1, 9]])},
                "source_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_source(data, vocab)

    assert len(actual) == 2
    assert "source_input_features" in actual
    assert len(actual["source_input_features"]) == 1
    assert "tokens" in actual["source_input_features"]
    assert torch.all(
        actual["source_input_features"]["tokens"] 
            == expected["source_input_features"]["tokens"])
    assert "source_lengths" in actual
    assert torch.all(actual["source_lengths"] == expected["source_lengths"])

def test_batch_source_from_list_of_dict_of_list_of_strings():

    data = [
        {"tokens": ["A", "B", "C",], "positions": ["1", "2", "3"]},
        {"tokens": ["B", "D"], "positions": ["1", "2"]},
        {"tokens": ["X", "Y", "!", "Z"], "positions": ["1", "2", "3", "4"]}
    ]

    vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                  pad="<PAD>", unk="<UNK>", start="<START>")
    vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                  pad="<PAD>", unk="<UNK>", start="<START>")
    vocab = {"tokens": vocab1, "positions": vocab2}

    expected = {"source_input_features": {
                    "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                [2, 4, 6, 0, 0],
                                                [2, 7, 8, 1, 9]]),
                    "positions": torch.LongTensor([[2, 3, 4, 5, 0],
                                                   [2, 3, 4, 0, 0],
                                                   [2, 3, 4, 5, 6]])},
                "source_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_source(data, vocab)

    assert len(actual) == 2
    assert "source_input_features" in actual
    assert len(actual["source_input_features"]) == 2
    assert "tokens" in actual["source_input_features"]
    assert "positions" in actual["source_input_features"]
    assert torch.all(
        actual["source_input_features"]["tokens"] 
            == expected["source_input_features"]["tokens"])
    assert torch.all(
        actual["source_input_features"]["positions"] 
            == expected["source_input_features"]["positions"])
    assert "source_lengths" in actual
    assert torch.all(actual["source_lengths"] == expected["source_lengths"])

def test_batch_source_from_list_of_dict_of_strings():

    data = [
        {"tokens": "A B C", "positions": "1 2 3"},
        {"tokens": "B D", "positions": "1 2"},
        {"tokens": "X Y ! Z", "positions": "1 2 3 4"}
    ]

    vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                  pad="<PAD>", unk="<UNK>", start="<START>")
    vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                  pad="<PAD>", unk="<UNK>", start="<START>")
    vocab = {"tokens": vocab1, "positions": vocab2}

    expected = {"source_input_features": {
                    "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                [2, 4, 6, 0, 0],
                                                [2, 7, 8, 1, 9]]),
                    "positions": torch.LongTensor([[2, 3, 4, 5, 0],
                                                   [2, 3, 4, 0, 0],
                                                   [2, 3, 4, 5, 6]])},
                "source_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_source(data, vocab)

    assert len(actual) == 2
    assert "source_input_features" in actual
    assert len(actual["source_input_features"]) == 2
    assert "tokens" in actual["source_input_features"]
    assert "positions" in actual["source_input_features"]
    assert torch.all(
        actual["source_input_features"]["tokens"] 
            == expected["source_input_features"]["tokens"])
    assert torch.all(
        actual["source_input_features"]["positions"] 
            == expected["source_input_features"]["positions"])
    assert "source_lengths" in actual
    assert torch.all(actual["source_lengths"] == expected["source_lengths"])


def test_batch_target_list_of_list_of_strings():

    vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                 pad="<PAD>", unk="<UNK>", start="<START>",
                                 stop="<STOP>")
    data = [["A", "B", "C",],
            ["B", "D"],
            ["X", "Y", "!", "Z"]]

    expected = {"target_input_features": {
                    "tokens": torch.LongTensor([[2, 4, 5, 6, 0],
                                                [2, 5, 7, 0, 0],
                                                [2, 8, 9, 1, 10]])},
                "target_output_features": {
                    "tokens": torch.LongTensor([[4, 5, 6, 3, 0],
                                                [5, 7, 3, 0, 0],
                                                [8, 9, 1, 10, 3]])},

                "target_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_target(
        data, vocab)

    assert len(actual) == 3
    assert "target_input_features" in actual
    assert "target_output_features" in actual
    assert len(actual["target_input_features"]) == 1
    assert len(actual["target_output_features"]) == 1
    assert "tokens" in actual["target_input_features"]
    assert "tokens" in actual["target_output_features"]
    assert torch.all(
        actual["target_input_features"]["tokens"] 
            == expected["target_input_features"]["tokens"])
    assert torch.all(
        actual["target_output_features"]["tokens"] 
            == expected["target_output_features"]["tokens"])
    assert "target_lengths" in actual
    assert torch.all(actual["target_lengths"] == expected["target_lengths"])

def test_batch_target_list_of_strings():

    vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                 pad="<PAD>", unk="<UNK>", start="<START>",
                                 stop="<STOP>")
    data = ["A B C",
            "B D",
            "X Y ! Z"]

    expected = {"target_input_features": {
                    "tokens": torch.LongTensor([[2, 4, 5, 6, 0],
                                                [2, 5, 7, 0, 0],
                                                [2, 8, 9, 1, 10]])},
                "target_output_features": {
                    "tokens": torch.LongTensor([[4, 5, 6, 3, 0],
                                                [5, 7, 3, 0, 0],
                                                [8, 9, 1, 10, 3]])},

                "target_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_target(data, vocab)

    assert len(actual) == 3
    assert "target_input_features" in actual
    assert "target_output_features" in actual
    assert len(actual["target_input_features"]) == 1
    assert len(actual["target_output_features"]) == 1
    assert "tokens" in actual["target_input_features"]
    assert "tokens" in actual["target_output_features"]
    assert torch.all(
        actual["target_input_features"]["tokens"] 
            == expected["target_input_features"]["tokens"])
    assert torch.all(
        actual["target_output_features"]["tokens"] 
            == expected["target_output_features"]["tokens"])
    assert "target_lengths" in actual
    assert torch.all(actual["target_lengths"] == expected["target_lengths"])

def test_batch_target_from_list_of_dict_of_list_of_strings():

    data = [
        {"tokens": ["A", "B", "C",], "positions": ["1", "2", "3"]},
        {"tokens": ["B", "D"], "positions": ["1", "2"]},
        {"tokens": ["X", "Y", "!", "Z"], "positions": ["1", "2", "3", "4"]}
    ]

    vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                  pad="<PAD>", unk="<UNK>", start="<START>",
                                  stop="<STOP>")
    vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                  pad="<PAD>", unk="<UNK>", start="<START>",
                                  stop="<STOP>")
    vocab = {"tokens": vocab1, "positions": vocab2}

    expected = {"target_input_features": {
                    "tokens": torch.LongTensor([[2, 4, 5, 6, 0],
                                                [2, 5, 7, 0, 0],
                                                [2, 8, 9, 1, 10]]),
                    "positions": torch.LongTensor([[2, 4, 5, 6, 0],
                                                   [2, 4, 5, 0, 0],
                                                   [2, 4, 5, 6, 7]])},
                "target_output_features": {
                    "tokens": torch.LongTensor([[4, 5, 6, 3, 0],
                                                [5, 7, 3, 0, 0],
                                                [8, 9, 1, 10, 3]]),
                    "positions": torch.LongTensor([[4, 5, 6, 3, 0],
                                                   [4, 5, 3, 0, 0],
                                                   [4, 5, 6, 7, 3]])},
                "target_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_target(data, vocab)

    assert len(actual) == 3
    assert "target_input_features" in actual
    assert "target_output_features" in actual
    assert len(actual["target_input_features"]) == 2
    assert len(actual["target_output_features"]) == 2
    assert "tokens" in actual["target_input_features"]
    assert "positions" in actual["target_input_features"]
    assert "tokens" in actual["target_output_features"]
    assert "positions" in actual["target_output_features"]
    assert torch.all(
        actual["target_input_features"]["tokens"] 
            == expected["target_input_features"]["tokens"])
    assert torch.all(
        actual["target_output_features"]["tokens"] 
            == expected["target_output_features"]["tokens"])
    assert torch.all(
        actual["target_input_features"]["positions"] 
            == expected["target_input_features"]["positions"])
    assert torch.all(
        actual["target_output_features"]["positions"] 
            == expected["target_output_features"]["positions"])
    assert "target_lengths" in actual
    assert torch.all(actual["target_lengths"] == expected["target_lengths"])

def test_batch_target_from_list_of_dict_of_strings():

    data = [
        {"tokens": "A B C", "positions": "1 2 3"},
        {"tokens": "B D", "positions": "1 2"},
        {"tokens": "X Y ! Z", "positions": "1 2 3 4"}
    ]

    vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                  pad="<PAD>", unk="<UNK>", start="<START>",
                                  stop="<STOP>")
    vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                  pad="<PAD>", unk="<UNK>", start="<START>",
                                  stop="<STOP>")
    vocab = {"tokens": vocab1, "positions": vocab2}

    expected = {"target_input_features": {
                    "tokens": torch.LongTensor([[2, 4, 5, 6, 0],
                                                [2, 5, 7, 0, 0],
                                                [2, 8, 9, 1, 10]]),
                    "positions": torch.LongTensor([[2, 4, 5, 6, 0],
                                                   [2, 4, 5, 0, 0],
                                                   [2, 4, 5, 6, 7]])},
                "target_output_features": {
                    "tokens": torch.LongTensor([[4, 5, 6, 3, 0],
                                                [5, 7, 3, 0, 0],
                                                [8, 9, 1, 10, 3]]),
                    "positions": torch.LongTensor([[4, 5, 6, 3, 0],
                                                   [4, 5, 3, 0, 0],
                                                   [4, 5, 6, 7, 3]])},
                "target_lengths": torch.LongTensor([4, 3, 5])}

    actual = seq2seq_batcher.batch_target(data, vocab)

    assert len(actual) == 3
    assert "target_input_features" in actual
    assert "target_output_features" in actual
    assert len(actual["target_input_features"]) == 2
    assert len(actual["target_output_features"]) == 2
    assert "tokens" in actual["target_input_features"]
    assert "positions" in actual["target_input_features"]
    assert "tokens" in actual["target_output_features"]
    assert "positions" in actual["target_output_features"]
    assert torch.all(
        actual["target_input_features"]["tokens"] 
            == expected["target_input_features"]["tokens"])
    assert torch.all(
        actual["target_output_features"]["tokens"] 
            == expected["target_output_features"]["tokens"])
    assert torch.all(
        actual["target_input_features"]["positions"] 
            == expected["target_input_features"]["positions"])
    assert torch.all(
        actual["target_output_features"]["positions"] 
            == expected["target_output_features"]["positions"])
    assert "target_lengths" in actual
    assert torch.all(actual["target_lengths"] == expected["target_lengths"])

def test_batch_copy_alignment_list_of_list_of_strings():

    src_data = [["E1", "E2", "C", "D"],
                ["A", "B", "R1", "C"],
                ["B", "A", "7"]]

    tgt_data = [["1", "E2", "2", "3"],
                ["4", "5", "?"],
                ["6", "7", "8", "7"]]
    alignment = [[-1, 1, -1, -1],
                 [-1, -1, 2],
                 [-1,-1, -1, -1]]
    vocab = Vocab.from_word_list(
        ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
        pad="<PAD>", unk="<UNK>", start="<START>", stop="<STOP>")

    actual1 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment)

    expected_copy_probs1 = torch.FloatTensor(
        [[0, .5,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])

    expected_copy_tgts1 = torch.LongTensor(
        [[-1,  1, -1, -1, -1],
         [-1, -1,  6, -1, -1],
         [-1, -1, -1, -1, -1]])

    expected_src_vcb_map1 = torch.FloatTensor(
        [[[0,0,0, 0,0,0, 0,0],
          [1,0,0, 0,0,0, 0,0],
          [0,1,0, 0,0,0, 0,0],
          [0,0,1, 0,0,0, 0,0],
          [0,0,0, 1,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,0,0, 1,0],
          [0,0,1, 0,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,0, 0,1],
          [0,0,0, 0,0,0, 0,0]]])

    assert torch.all(actual1["source_vocab_map"] == expected_src_vcb_map1)
    assert torch.all(actual1["copy_probabilty"] == expected_copy_probs1)
    assert torch.all(actual1["copy_targets"] == expected_copy_tgts1)

    actual2 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment, mixture_copy_prob=.6)

    expected_copy_probs2 = torch.FloatTensor(
        [[0, .6,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])
    assert torch.all(actual2["copy_probabilty"] == expected_copy_probs2)

def test_batch_copy_alignment_list_of_strings():

    src_data = ["E1 E2 C D",
                "A B R1 C",
                "B A 7"]

    tgt_data = ["1 E2 2 3",
                "4 5 ?",
                "6 7 8 7"]
    alignment = [[-1, 1, -1, -1],
                 [-1, -1, 2],
                 [-1,-1, -1, -1]]
    vocab = Vocab.from_word_list(
        ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
        pad="<PAD>", unk="<UNK>", start="<START>", stop="<STOP>")

    actual1 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment)

    expected_copy_probs1 = torch.FloatTensor(
        [[0, .5,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])

    expected_copy_tgts1 = torch.LongTensor(
        [[-1,  1, -1, -1, -1],
         [-1, -1,  6, -1, -1],
         [-1, -1, -1, -1, -1]])

    expected_src_vcb_map1 = torch.FloatTensor(
        [[[0,0,0, 0,0,0, 0,0],
          [1,0,0, 0,0,0, 0,0],
          [0,1,0, 0,0,0, 0,0],
          [0,0,1, 0,0,0, 0,0],
          [0,0,0, 1,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,0,0, 1,0],
          [0,0,1, 0,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,0, 0,1],
          [0,0,0, 0,0,0, 0,0]]])

    assert torch.all(actual1["source_vocab_map"] == expected_src_vcb_map1)
    assert torch.all(actual1["copy_probabilty"] == expected_copy_probs1)
    assert torch.all(actual1["copy_targets"] == expected_copy_tgts1)

    actual2 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment, mixture_copy_prob=.6)

    expected_copy_probs2 = torch.FloatTensor(
        [[0, .6,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])
    assert torch.all(actual2["copy_probabilty"] == expected_copy_probs2)

def test_batch_copy_alignment_list_of_dict_of_list_of_strings():

    src_data = [
        {"tokens": ["E1", "E2", "C", "D"]},
        {"tokens": ["A", "B", "R1", "C"]},
        {"tokens": ["B", "A", "7"]}
    ]

    tgt_data = [
        {"tokens": ["1", "E2", "2", "3"]},
        {"tokens": ["4", "5", "?"]},
        {"tokens": ["6", "7", "8", "7"]}
    ]

    alignment = [[-1, 1, -1, -1],
                 [-1, -1, 2],
                 [-1,-1, -1, -1]]

    vocab = Vocab.from_word_list(
        ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
        pad="<PAD>", unk="<UNK>", start="<START>", stop="<STOP>")

    actual1 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment)

    expected_copy_probs1 = torch.FloatTensor(
        [[0, .5,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])

    expected_copy_tgts1 = torch.LongTensor(
        [[-1,  1, -1, -1, -1],
         [-1, -1,  6, -1, -1],
         [-1, -1, -1, -1, -1]])

    expected_src_vcb_map1 = torch.FloatTensor(
        [[[0,0,0, 0,0,0, 0,0],
          [1,0,0, 0,0,0, 0,0],
          [0,1,0, 0,0,0, 0,0],
          [0,0,1, 0,0,0, 0,0],
          [0,0,0, 1,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,0,0, 1,0],
          [0,0,1, 0,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,0, 0,1],
          [0,0,0, 0,0,0, 0,0]]])

    assert torch.all(actual1["source_vocab_map"] == expected_src_vcb_map1)
    assert torch.all(actual1["copy_probabilty"] == expected_copy_probs1)
    assert torch.all(actual1["copy_targets"] == expected_copy_tgts1)

    actual2 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment, mixture_copy_prob=.6)

    expected_copy_probs2 = torch.FloatTensor(
        [[0, .6,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])
    assert torch.all(actual2["copy_probabilty"] == expected_copy_probs2)

def test_batch_copy_alignment_list_of_dict_of_strings():

    src_data = [
        {"tokens": "E1 E2 C D"},
        {"tokens": "A B R1 C"},
        {"tokens": "B A 7"}
    ]

    tgt_data = [
        {"tokens": "1 E2 2 3"},
        {"tokens": "4 5 ?"},
        {"tokens": "6 7 8 7"}
    ]

    alignment = [[-1, 1, -1, -1],
                 [-1, -1, 2],
                 [-1,-1, -1, -1]]

    vocab = Vocab.from_word_list(
        ["1", "2", "3", "4", "5", "6", "7", "8", "E2"],
        pad="<PAD>", unk="<UNK>", start="<START>", stop="<STOP>")

    actual1 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment)

    expected_copy_probs1 = torch.FloatTensor(
        [[0, .5,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])

    expected_copy_tgts1 = torch.LongTensor(
        [[-1,  1, -1, -1, -1],
         [-1, -1,  6, -1, -1],
         [-1, -1, -1, -1, -1]])

    expected_src_vcb_map1 = torch.FloatTensor(
        [[[0,0,0, 0,0,0, 0,0],
          [1,0,0, 0,0,0, 0,0],
          [0,1,0, 0,0,0, 0,0],
          [0,0,1, 0,0,0, 0,0],
          [0,0,0, 1,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,0,0, 1,0],
          [0,0,1, 0,0,0, 0,0]],
         [[0,0,0, 0,0,0, 0,0],
          [0,0,0, 0,0,1, 0,0],
          [0,0,0, 0,1,0, 0,0],
          [0,0,0, 0,0,0, 0,1],
          [0,0,0, 0,0,0, 0,0]]])

    assert torch.all(actual1["source_vocab_map"] == expected_src_vcb_map1)
    assert torch.all(actual1["copy_probabilty"] == expected_copy_probs1)
    assert torch.all(actual1["copy_targets"] == expected_copy_tgts1)

    actual2 = seq2seq_batcher.batch_copy_alignments(
        src_data, tgt_data, vocab, alignment, mixture_copy_prob=.6)

    expected_copy_probs2 = torch.FloatTensor(
        [[0, .6,  0, 0,  0],
         [0,  0,  1, 0, -1],
         [0,  0,  0, 0,  0]])
    assert torch.all(actual2["copy_probabilty"] == expected_copy_probs2)
