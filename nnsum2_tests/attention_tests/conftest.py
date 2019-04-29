import pytest
import torch
import random


@pytest.fixture(scope="module")
def batch_size():
    return 3

@pytest.fixture(scope="module")
def key_length():
    return 5

@pytest.fixture(scope="module")
def hidden_size():
    return 16

@pytest.fixture(scope="module")
def query_length():
    return 4

@pytest.fixture(scope="function")
def key(batch_size, key_length, hidden_size):
    return torch.nn.Parameter(
        torch.FloatTensor(batch_size, key_length, hidden_size).normal_())

@pytest.fixture(scope="function", params=[True, False], 
                ids=["key_mask", "no_key_mask"])
def key_mask(request, batch_size, key_length):
    use_mask = request.param
    if use_mask:
        key_mask = torch.ByteTensor(batch_size, key_length).fill_(0)
        lengths = list(range(1, key_length + 1))[-batch_size:]
        random.seed(0)
        random.shuffle(lengths)
        for batch, length in enumerate(lengths):
            key_mask[batch, length:].fill_(1)
        return key_mask
    else:
        return None

@pytest.fixture(scope="function")
def query(batch_size, query_length, hidden_size):
    return torch.nn.Parameter(
        torch.FloatTensor(query_length, batch_size, hidden_size).normal_())

@pytest.fixture(scope="function", params=[True, False],
                ids=["init_state", "no_init_state"])
def init_accumulator(request, batch_size, key_length, key_mask):
    use_state = request.param
    if use_state:
        logits = torch.FloatTensor(batch_size, 1, key_length).normal_()
        if key_mask is not None:
            logits = logits.masked_fill(key_mask.unsqueeze(1), float("-inf"))
        accumulator = torch.softmax(logits, dim=2)
        return {"accumulator": torch.nn.Parameter(accumulator)}
    else:
        return None

