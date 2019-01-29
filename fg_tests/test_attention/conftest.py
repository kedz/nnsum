import pytest
import torch


@pytest.fixture(scope="module")
def batch_size():
    return 4

@pytest.fixture(scope="module")
def context_len():
    return 6

@pytest.fixture(scope="module")
def hidden_size():
    return 15

@pytest.fixture(scope="module")
def query_len():
    return 3

@pytest.fixture(scope="module")
def context(batch_size, context_len, hidden_size):
    return torch.FloatTensor(batch_size, context_len, hidden_size).normal_()

@pytest.fixture(scope="module")
def query(batch_size, query_len, hidden_size):
    return torch.FloatTensor(query_len, batch_size, hidden_size).normal_()




