import pytest
import torch
from nnsum.util import batch_pad_and_stack_matrix


def data_param_gen():
    return [{"target_texts": "simple"}]

def data_param_names(params):
    return "tgt_texts={target_texts}".format(
        **params)

@pytest.fixture(scope="module", params=data_param_gen(), ids=data_param_names)
def data_params(request):
    return request.param

def decoder_param_gen():
    return [
        {"attention": "none", "rnn_cell": "gru", "copy_attention": "none"},
        {"attention": "none", "rnn_cell": "lstm", "copy_attention": "none"}, 
        {"attention": "dot", "rnn_cell": "gru", "copy_attention": "none"},
        {"attention": "dot", "rnn_cell": "lstm", "copy_attention": "none"},
    ]

def decoder_param_names(params):
    return ("attn={attention}:cpy_attn={copy_attention}"
            ":rcell={rnn_cell}").format(**params)

@pytest.fixture(scope="module", params=decoder_param_gen(), 
                ids=decoder_param_names)
def decoder_params(request):
    return request.param

@pytest.fixture(scope="module")
def gold_outputs(eval_data):
    return eval_data["target_output_features"]["tokens"]

@pytest.fixture(scope="module")
def output_mask(gold_outputs, vocab):
    return gold_outputs.eq(vocab.pad_index)

@pytest.fixture(scope="module")
def batch_state(decoder, eval_data, encoder_state, context):
    return decoder.next_state(
        encoder_state, 
        inputs=eval_data["target_input_features"],
        context=context,
        compute_log_probability=True)

@pytest.fixture(scope="module")
def singleton_eval_data(eval_data, batch_size):
    singletons = []
    for i in range(batch_size):
        tgt_len = eval_data["target_lengths"][i:i+1]
        tgt_in = eval_data["target_input_features"]["tokens"]
        tgt_in = tgt_in[i:i+1,:tgt_len[0]]
        tgt_out = eval_data["target_output_features"]["tokens"]
        tgt_out = tgt_in[i:i+1,:tgt_len[0]]
        singletons.append({
            "target_input_features": {"tokens": tgt_in},
            "target_output_features": {"tokens": tgt_out},
            "target_lengths": tgt_len})
    return singletons

@pytest.fixture(scope="module")
def singleton_states(decoder_params, decoder, singleton_eval_data, 
                     encoder_state, context, batch_size):
    states = []
    for i in range(batch_size):
        if decoder_params["rnn_cell"] == "lstm":
            encoder_state_i = encoder_state.reindex[:,i:i+1]
        else:
            encoder_state_i = encoder_state[:,i:i+1]
        inputs_i = singleton_eval_data[i]["target_input_features"]
        context_i = context[i:i+1]
        state = decoder.next_state(
            encoder_state_i, 
            inputs=inputs_i,
            context=context_i,
            compute_log_probability=True)
        states.append(state)
    return states

@pytest.fixture(scope="module")
def batch_log_probability(batch_state, output_mask):
    lp = batch_state["log_probability"]
    return lp.masked_fill(output_mask.t().unsqueeze(-1), 0.)

@pytest.fixture(scope="module")
def batch_predictions(batch_log_probability):
    return batch_log_probability.permute(1,0,2).max(2)[1]

@pytest.fixture(scope="module")
def batch_rnn_outputs(batch_state, output_mask):
    output = batch_state["rnn_outputs"]
    return output.masked_fill(output_mask.t().unsqueeze(-1), 0.)

@pytest.fixture(scope="module")
def batch_logits(batch_state, output_mask):
    output = batch_state["logits"]
    return output.masked_fill(output_mask.t().unsqueeze(-1), 0.)

@pytest.fixture(scope="module")
def batch_context_attention(batch_state, output_mask):
    if "context_attention" not in batch_state:
        return None
    else:
        output = batch_state["context_attention"]
        return output.masked_fill(output_mask.t().unsqueeze(-1), 0.)

def tensor_equal(a, b):
    if a.dtype == torch.int64:
        assert torch.all(a == b)
    else:
        assert torch.allclose(a, b, atol=1e-5)

@pytest.fixture(scope="module")
def singleton_log_probability(singleton_states, output_mask):
    lps = []
    for state in singleton_states:
        lps.append(state["log_probability"].squeeze(1))
    lps = batch_pad_and_stack_matrix(lps, 0).permute(1, 0, 2)
    return lps

@pytest.fixture(scope="module")
def singleton_logits(singleton_states, output_mask):
    logits = []
    for state in singleton_states:
        logits.append(state["logits"].squeeze(1))
    logits = batch_pad_and_stack_matrix(logits, 0).permute(1, 0, 2)
    return logits

@pytest.fixture(scope="module")
def singleton_predictions(singleton_log_probability):
    return singleton_log_probability.max(2)[1].t()

@pytest.fixture(scope="module")
def singleton_rnn_outputs(singleton_states):
    outputs = []
    for state in singleton_states:
        outputs.append(state["rnn_outputs"].squeeze(1))
    return batch_pad_and_stack_matrix(outputs, 0.).permute(1, 0, 2)

@pytest.fixture(scope="module")
def singleton_context_attention(singleton_states):
    if "context_attention" not in singleton_states[0]:
        return None
    else:
        outputs = []
        for state in singleton_states:
            outputs.append(state["context_attention"].squeeze(1))
        return batch_pad_and_stack_matrix(outputs, 0.).permute(1, 0, 2)

def test_batch_predictions(batch_predictions, gold_outputs):
    tensor_equal(batch_predictions, gold_outputs)

def test_singleton_predictions(singleton_predictions, gold_outputs):
    tensor_equal(singleton_predictions, gold_outputs)

def test_rnn_outputs(singleton_rnn_outputs, batch_rnn_outputs):
    tensor_equal(singleton_rnn_outputs, batch_rnn_outputs)

def test_log_probability(singleton_log_probability, batch_log_probability):
    tensor_equal(singleton_log_probability, batch_log_probability)

def test_logits(singleton_logits, batch_logits):
    tensor_equal(singleton_logits, batch_logits)

def test_context_attention(singleton_context_attention, 
                           batch_context_attention):
    if singleton_context_attention is None and batch_context_attention is None:
        return
    tensor_equal(singleton_context_attention, batch_context_attention)

@pytest.mark.xfail
def test_logits_backprop(singleton_logits, batch_logits):
    pass
