import pytest
import torch
import nnsum.data.seq2seq_batcher as seq2seq_batcher
from nnsum.embedding_context import Vocab


@pytest.fixture(
    scope="module",
    params=["list_of_list_of_strings", "list_of_strings", 
            "list_of_dict_of_list_of_strings", "list_of_dict_of_strings"])
def input_mode(request):
    return request.param

@pytest.fixture(scope="module")
def source_batch(input_mode):
    if input_mode == "list_of_list_of_strings":
        vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                     pad="<PAD>", unk="<UNK>", start="<START>")
        data = [["A", "B", "C",],
                ["B", "D"],
                ["X", "Y", "!", "Z"]]
    elif input_mode == "list_of_strings":
        vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                     pad="<PAD>", unk="<UNK>", start="<START>")
        data = ["A B C",
                "B D",
                "X Y ! Z"]

    elif input_mode == "list_of_dict_of_list_of_strings":
        data = [
            {"tokens": ["A", "B", "C",], "positions": ["1", "2", "3"]},
            {"tokens": ["B", "D"], "positions": ["1", "2"]},
            {"tokens": ["X", "Y", "!", "Z"], "positions": ["1", "2", "3", "4"]}
        ]

        vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>")
        vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>")
        vocab = {"tokens": vocab1, "positions": vocab2}

    elif input_mode == "list_of_dict_of_strings":
        data = [
            {"tokens": "A B C", "positions": "1 2 3"},
            {"tokens": "B D", "positions": "1 2"},
            {"tokens": "X Y ! Z", "positions": "1 2 3 4"}
        ]
        
        vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>")
        vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>")
        vocab = {"tokens": vocab1, "positions": vocab2}

    else:
        raise ValueError("Invalid parameter: {}".format(request.param))


    return seq2seq_batcher.batch_source(data, vocab)

@pytest.fixture(scope="module")
def expected_source_batch(input_mode):
    if input_mode in ["list_of_list_of_strings", "list_of_strings"]:
        return {"source_input_features": {
                        "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                    [2, 4, 6, 0, 0],
                                                    [2, 7, 8, 1, 9]])},
                "source_lengths": torch.LongTensor([4, 3, 5]),
                "source_mask": torch.ByteTensor([[0,0,0,0,1],
                                                 [0,0,0,1,1],
                                                 [0,0,0,0,0]])}
    else:
        return {"source_input_features": {
                    "tokens": torch.LongTensor([[2, 3, 4, 5, 0],
                                                [2, 4, 6, 0, 0],
                                                [2, 7, 8, 1, 9]]),
                    "positions": torch.LongTensor([[2, 3, 4, 5, 0],
                                                   [2, 3, 4, 0, 0],
                                                   [2, 3, 4, 5, 6]])},
                "source_lengths": torch.LongTensor([4, 3, 5]),
                "source_mask": torch.ByteTensor([[0,0,0,0,1],
                                                 [0,0,0,1,1],
                                                 [0,0,0,0,0]])}

def test_source_batch(source_batch, expected_source_batch):
    assert len(source_batch) == len(expected_source_batch)
    assert "source_input_features" in source_batch
    assert len(source_batch["source_input_features"]) == \
        len(expected_source_batch["source_input_features"])

    for name in source_batch["source_input_features"].keys():
        assert torch.all(
            source_batch["source_input_features"][name] ==
                expected_source_batch["source_input_features"][name])
            
    assert "source_lengths" in source_batch
    assert torch.all(
        source_batch["source_lengths"] == 
            expected_source_batch["source_lengths"])

    assert "source_mask" in source_batch
    assert torch.all(
        source_batch["source_mask"] == expected_source_batch["source_mask"])

@pytest.fixture(scope="module")
def target_batch(input_mode):
    if input_mode == "list_of_list_of_strings":
        vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                     pad="<PAD>", unk="<UNK>", start="<START>",
                                     stop="<STOP>")
        data = [["A", "B", "C",],
                ["B", "D"],
                ["X", "Y", "!", "Z"]]
    elif input_mode == "list_of_strings":
        vocab = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                     pad="<PAD>", unk="<UNK>", start="<START>",
                                     stop="<STOP>")
        data = ["A B C",
                "B D",
                "X Y ! Z"]

    elif input_mode == "list_of_dict_of_list_of_strings":
        data = [
            {"tokens": ["A", "B", "C",], "positions": ["1", "2", "3"]},
            {"tokens": ["B", "D"], "positions": ["1", "2"]},
            {"tokens": ["X", "Y", "!", "Z"], "positions": ["1", "2", "3", "4"]}
        ]

        vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>", stop="<STOP>")
        vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>", stop="<STOP>")
        vocab = {"tokens": vocab1, "positions": vocab2}

    elif input_mode == "list_of_dict_of_strings":
        data = [
            {"tokens": "A B C", "positions": "1 2 3"},
            {"tokens": "B D", "positions": "1 2"},
            {"tokens": "X Y ! Z", "positions": "1 2 3 4"}
        ]
        
        vocab1 = Vocab.from_word_list(["A", "B", "C", "D", "X", "Y", "Z"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>", stop="<STOP>")
        vocab2 = Vocab.from_word_list(["1", "2", "3", "4"],
                                      pad="<PAD>", unk="<UNK>", 
                                      start="<START>", stop="<STOP>")
        vocab = {"tokens": vocab1, "positions": vocab2}

    else:
        raise ValueError("Invalid parameter: {}".format(request.param))

    return seq2seq_batcher.batch_target(data, vocab)

@pytest.fixture(scope="module")
def expected_target_batch(input_mode):
    if input_mode in ["list_of_list_of_strings", "list_of_strings"]:
        return {"target_input_features": {
                    "tokens": torch.LongTensor([[2, 4, 5, 6, 0],
                                                [2, 5, 7, 0, 0],
                                                [2, 8, 9, 1, 10]])},
                "target_output_features": {
                    "tokens": torch.LongTensor([[4, 5, 6, 3, 0],
                                                [5, 7, 3, 0, 0],
                                                [8, 9, 1, 10, 3]])},
                "target_lengths": torch.LongTensor([4, 3, 5])}
    else:
        return {"target_input_features": {
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

def test_target_batch(target_batch, expected_target_batch):
    assert len(target_batch) == len(expected_target_batch)

    assert "target_input_features" in target_batch
    assert len(target_batch["target_input_features"]) == \
        len(expected_target_batch["target_input_features"])
    for name in target_batch["target_input_features"].keys():
        assert torch.all(
            target_batch["target_input_features"][name] ==
                expected_target_batch["target_input_features"][name])

    assert "target_output_features" in target_batch
    assert len(target_batch["target_output_features"]) == \
        len(expected_target_batch["target_output_features"])
    for name in target_batch["target_output_features"].keys():
        assert torch.all(
            target_batch["target_output_features"][name] ==
                expected_target_batch["target_output_features"][name])
         
    assert "target_lengths" in target_batch
    assert torch.all(
        target_batch["target_lengths"] == 
            expected_target_batch["target_lengths"])
