import pytest
import torch
import nnsum2.torch as ntorch
from nnsum2.seq2seq.searches import GreedySearch


def output2inputs(output, target_vocab):
    """
    Convert search output into input for forward method.
    """
    tgt_in = torch.cat(
        [output.new(output.size(0), 1).fill_(target_vocab.start_index),
         output[:,:-1]],
        dim=1)
    tgt_in = tgt_in.masked_fill(
        tgt_in.eq(target_vocab.stop_index), target_vocab.pad_index)

    return tgt_in

def mask_outputs(greedy_output, forward_output, vocab):
    for b, row in enumerate(greedy_output):
        row = row.tolist()
        if vocab.stop_index not in row:
            continue
        idx = row.index(vocab.stop_index) + 1
        forward_output.data[b,idx:].fill_(vocab.pad_index)

RNN_ATTN_TYPES = ["none", "bilinear", "accum_bilinear"]
RNN_PG_ATTN_TYPES = ["bilinear", "accum_bilinear"]

RNN_RESULT_TYPES = [
    {"field": "rnn_input"},
    {"field": "rnn_output"},
    {"field": "context_attention"},
    {"field": "attention_output"},
    {"field": "target_logits"},
    {"field": "output"},
    {"field": "log_probability"},
    {"field": "context_attention_state", "final_state": True},
    {"field": "decoder_state", "final_state": True},
]

RNN_PG_RESULT_TYPES = [
    {"field": "rnn_input"},
    {"field": "rnn_output"},
    {"field": "context_attention"},
    {"field": "copy_switch"},
    {"field": "context_attention"},
    {"field": "attention_output"},
    {"field": "output"},
    {"field": "context_attention_state", "final_state": True},
    {"field": "decoder_state", "final_state": True},
    {"field": "generator_logits"},
    {"field": "generator_distribution"},
    {"field": "pointer_distribution"},
    {"field": "generator_probability"},
    {"field": "pointer_probability"},
    {"field": "log_probability"},
]

def param_generator():
    decoder_types = ["rnn", "rnn_pointer_generator"]
    dec_attn_types = {
        "rnn": RNN_ATTN_TYPES,
        "rnn_pointer_generator": RNN_PG_ATTN_TYPES,
    }
    dec_result_types = {
        "rnn": RNN_RESULT_TYPES,
        "rnn_pointer_generator": RNN_PG_RESULT_TYPES,
    }
    
    for dec_type in decoder_types:
        for attn in dec_attn_types[dec_type]:
            for result in dec_result_types[dec_type]:
                yield {
                    "decoder_params": {
                        "type": dec_type,
                        "attention_mechanism": attn,
                    },
                    "result_params": result,
                }

def param_name(params):
    return params["decoder_params"]["type"] + "-" + \
        params["decoder_params"]["attention_mechanism"] + "-" + \
        params["result_params"]["field"]
@pytest.mark.parametrize("test_params", param_generator(), ids=param_name)
def test_consistent_with_forward(test_params, decoder_factory, 
                                 init_rnn_state, context,
                                 target_vocab):

    decoder = decoder_factory(**test_params["decoder_params"])
    
    search = GreedySearch(decoder, {"encoder_state": init_rnn_state},
                          context, max_steps=10, return_incomplete=True,
                          compute_log_probability=True)
    greedy_output = search.get_result("output")
    batch_inputs = output2inputs(greedy_output, target_vocab)

    result_field = test_params["result_params"]["field"]

    forward_state = decoder(
        init_rnn_state, batch_inputs, context, None, 
        compute_output=True, compute_log_probability=True)

    # Forward algo doesn't mask outputs but search does.
    mask_outputs(greedy_output, forward_state["output"], target_vocab)

    greedy_result = search.get_result(result_field)
    dim_names = search._states._dim_names[result_field]
    
    if dim_names is None:
        assert forward_state[result_field] is None
        assert search.get_result(result_field) is None
    else:
        seq_idx = dim_names.index("sequence")
        batch_idx = dim_names.index("batch")
        forward_result = forward_state[result_field]

        if test_params["result_params"].get("final_state", False):
            greedy_result = greedy_result.narrow(seq_idx, -1, 1).squeeze()
            forward_result = forward_result.squeeze()

        assert ntorch.allclose(greedy_result, forward_result, 
                               rtol=0, atol=1e-5)

GRAD_RNN_ATTN_TYPES = ["none", "bilinear", "accum_bilinear"]
GRAD_RNN_PG_ATTN_TYPES = ["bilinear", "accum_bilinear"]

GRAD_RNN_RESULT_TYPES = [
#    {"field": "rnn_input"},
#    {"field": "rnn_output"},
#    {"field": "context_attention"},
#    {"field": "attention_output"},
#    {"field": "target_logits"},
#    {"field": "output"},
    {"field": "log_probability"},
#    {"field": "context_attention_state", "final_state": True},
#    {"field": "decoder_state", "final_state": True},
]

GRAD_RNN_PG_RESULT_TYPES = [
#    {"field": "rnn_input"},
#    {"field": "rnn_output"},
#    {"field": "context_attention"},
#    {"field": "copy_switch"},
#    {"field": "context_attention"},
#    {"field": "attention_output"},
#    {"field": "output"},
#    {"field": "context_attention_state", "final_state": True},
#    {"field": "decoder_state", "final_state": True},
#    {"field": "generator_logits"},
#    {"field": "generator_distribution"},
#    {"field": "pointer_distribution"},
#    {"field": "generator_probability"},
#    {"field": "pointer_probability"},
    {"field": "log_probability"},
]

def grad_param_generator():
    decoder_types = ["rnn", "rnn_pointer_generator"]
    dec_attn_types = {
        "rnn": GRAD_RNN_ATTN_TYPES,
        "rnn_pointer_generator": GRAD_RNN_PG_ATTN_TYPES,
    }
    dec_result_types = {
        "rnn": GRAD_RNN_RESULT_TYPES,
        "rnn_pointer_generator": GRAD_RNN_PG_RESULT_TYPES,
    }
    
    for dec_type in decoder_types:
        for attn in dec_attn_types[dec_type]:
            for result in dec_result_types[dec_type]:
                yield {
                    "decoder_params": {
                        "type": dec_type,
                        "attention_mechanism": attn,
                    },
                    "result_params": result,
                }

@pytest.mark.parametrize("test_params", grad_param_generator(), ids=param_name)
def test_grad_consistent_with_forward(test_params, decoder_factory, 
                                      init_rnn_state, context,
                                      target_vocab):

    decoder = decoder_factory(**test_params["decoder_params"])
    
    search = GreedySearch(decoder, {"encoder_state": init_rnn_state},
                          context, max_steps=10, return_incomplete=True,
                          compute_log_probability=True)
    greedy_output = search.get_result("output")
    batch_inputs = output2inputs(greedy_output, target_vocab)

    result_field = test_params["result_params"]["field"]

    greedy_result = search.get_result(result_field)
    DUMMY_GRAD = greedy_result.clone().fill_(-1).detach()
    greedy_result.backward(gradient=DUMMY_GRAD)
    greedy_grads = {}
    for name, param in decoder.named_parameters():
        greedy_grads[name] = param.grad.clone()
        param.grad.fill_(0)
    greedy_grads["encoder_state"] = init_rnn_state.grad.clone() 
    init_rnn_state.grad.fill_(0)

    forward_state = decoder(
        init_rnn_state, batch_inputs, context, None, 
        compute_output=True, compute_log_probability=True)

    # Forward algo doesn't mask outputs but search does.
    mask_outputs(greedy_output, forward_state["output"], target_vocab)

    forward_result = forward_state[result_field]
    forward_result.backward(gradient=DUMMY_GRAD)
    forward_grads = {}
    for name, param in decoder.named_parameters():
        forward_grads[name] = param.grad.clone()
        param.grad.fill_(0)
    forward_grads["encoder_state"] = init_rnn_state.grad.clone() 
    init_rnn_state.grad.fill_(0)

    assert len(forward_grads) == len(greedy_grads)
    for name, fwd_grad in forward_grads.items():
        search_grad = greedy_grads[name]
        assert ntorch.allclose(search_grad, fwd_grad,
                               rtol=0, atol=1e-4)

