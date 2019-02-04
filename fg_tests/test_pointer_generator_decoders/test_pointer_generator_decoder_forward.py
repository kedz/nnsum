import pytest
import torch


def data_param_gen():
    return [{"sparse": False}]


def data_param_names(params):
    return ("sp={sparse}").format(**params)

@pytest.fixture(scope="module", params=data_param_gen(), 
                ids=data_param_names)
def data_params(request):
    return request.param

def decoder_param_gen():
    return [
        {"attention": "dot", "rnn_cell": "gru"},
        {"attention": "dot", "rnn_cell": "lstm"},
    ]

def decoder_param_names(params):
    return ("attn={attention}:rcell={rnn_cell}").format(**params)

@pytest.fixture(scope="module", params=decoder_param_gen(), 
                ids=decoder_param_names)
def decoder_params(request):
    return request.param

@pytest.fixture(scope="module")
def train_state(model, train_data):
    model.eval()
    return model.get_state(train_data, compute_log_probability=True)

def test_switch_probability(train_state, train_data):
    copy_mask = train_data["copy_probability"].t().eq(1)
    expected_copy = train_data["copy_probability"].t().masked_select(copy_mask)
    actual_copy = train_state["source_probability"].masked_select(copy_mask)
    assert torch.allclose(actual_copy, expected_copy, atol=1e-2)

    gen_mask = train_data["copy_probability"].t().eq(0)
    expected_gen = train_data["copy_probability"].t().masked_select(gen_mask)
    actual_gen = train_state["source_probability"].masked_select(gen_mask)
    assert torch.allclose(actual_gen, expected_gen, atol=1e-2)

    both_mask = train_data["copy_probability"].t().eq(.5)
    expected_both = train_data["copy_probability"].t().masked_select(both_mask)
    actual_both = (
            train_state["source_probability"].masked_select(both_mask) +
            train_state["target_probability"].masked_select(both_mask)
        ) / 2

    assert torch.allclose(actual_both, expected_both, atol=1e-2)

def test_copy_attention(train_state):
    assert train_state["copy_attention"].permute(1,0,2)[0,0].max(0)[1] == 1
    assert train_state["copy_attention"].permute(1,0,2)[0,2].max(0)[1] == 5
    assert train_state["copy_attention"].permute(1,0,2)[1,0].max(0)[1] == 1
    assert train_state["copy_attention"].permute(1,0,2)[2,0].max(0)[1] == 1

def test_log_probability(train_state, train_data):
    predictions = train_state["log_probability"].permute(1,0,2).max(2)[1]
    assert torch.all(predictions == train_data["copy_targets"])
