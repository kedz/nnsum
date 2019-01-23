import pytest
import torch
import nnsum.seq2seq as s2s


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
        {"attention": "none", "rnn_cell": "gru",},
        {"attention": "none", "rnn_cell": "lstm"}, 
        {"attention": "dot", "rnn_cell": "gru"},
        {"attention": "dot", "rnn_cell": "lstm"},
    ]

def decoder_param_names(params):
    return "attn={attention}:rcell={rnn_cell}".format(
        **params)

@pytest.fixture(scope="module", params=decoder_param_gen(), 
                ids=decoder_param_names)
def decoder_params(request):
    return request.param

@pytest.fixture(scope="module")
def greedy_state(decoder, encoder_state, context):
    return s2s.GreedySearch(decoder, encoder_state, context, max_steps=10)

@pytest.fixture(scope="module")
def eval_data(greedy_state, vocab, batch_size):
    tokens = greedy_state.get_result("outputs").t().view(3,-1)
    new_inputs = tokens.new(batch_size, tokens.size(1)).fill_(vocab.pad_index)
    new_inputs[:,0].fill_(vocab.start_index)
    new_outputs = tokens.new(batch_size, tokens.size(1)).fill_(vocab.pad_index)
    new_lengths = tokens.new(batch_size).long()
    for i, row in enumerate(tokens):
        try:
            length = row.tolist().index(vocab.stop_index) + 1
            new_inputs[i,1:length].copy_(row[:length-1])
            new_outputs[i,:length].copy_(row[:length])
            new_lengths[i] = length
        except ValueError:
            new_lengths[i] = 0
            continue
    return {"target_input_features": {"tokens": new_inputs},
            "target_output_features": {"tokens": new_outputs},
            "target_lengths": new_lengths}

@pytest.fixture(scope="module")
def eval_mask(vocab, eval_data, batch_size):
    mask = eval_data["target_input_features"]["tokens"].eq(vocab.pad_index)
    mask = mask.view(batch_size, -1)
    return mask

@pytest.fixture(scope="module")
def forward_state(decoder, eval_data, encoder_state, context, 
                  initialize_decoder):
    encoder_state, context = initialize_decoder(encoder_state, context)    
    fwd_state = decoder.next_state(
        encoder_state,
        inputs=eval_data["target_input_features"],
        context=context,
        compute_log_probability=True)
    return fwd_state

def get_forward_field(field, forward_state, batch_size, eval_mask):
    output = forward_state[field]
    steps = output.size(0)
    output = output.view(steps, batch_size, -1).masked_fill(
        eval_mask.t().unsqueeze(-1), 0.)
    return output

@pytest.fixture(scope="module")
def forward_rnn_outputs(forward_state, batch_size, eval_mask):
    return get_forward_field("rnn_outputs", forward_state, batch_size,
                             eval_mask)

@pytest.fixture(scope="module")
def forward_context_attention(forward_state, batch_size, eval_mask):
    if "context_attention" not in forward_state:
        return None
    else:
        return get_forward_field("context_attention", forward_state,
                                 batch_size, eval_mask)

@pytest.fixture(scope="module")
def forward_logits(forward_state, batch_size, eval_mask):
    return get_forward_field("logits", forward_state, batch_size,  
                             eval_mask)

@pytest.fixture(scope="module")
def forward_log_probability(forward_state, batch_size, eval_mask):
    return get_forward_field("log_probability", forward_state, batch_size,
                             eval_mask)

@pytest.fixture(scope="module")
def forward_output_log_probability(forward_log_probability, eval_data,
                                   batch_size):    
    tgts = eval_data["target_output_features"]["tokens"].t()
    steps = tgts.size(0)
    tgts = tgts.view(steps, batch_size, 1)
    return forward_log_probability.gather(2, tgts).squeeze(-1)

@pytest.fixture(scope="module")
def forward_outputs(forward_state, batch_size, eval_mask):
    output = forward_state["log_probability"].max(2)[1].t()
    steps = output.size(1)
    return output.view(batch_size, steps).masked_fill(eval_mask, 0)

@pytest.fixture(scope="module")
def forward_output_log_probability(forward_state, batch_size, eval_mask):
    output = forward_state["log_probability"].max(2)[0].t()
    steps = output.size(1)
    return output.view(batch_size, steps).masked_fill(eval_mask, 0)

@pytest.fixture(scope="module")
def greedy_rnn_outputs(greedy_state):
    return greedy_state.get_result("rnn_outputs")

@pytest.fixture(scope="module")
def greedy_logits(greedy_state):
    return greedy_state.get_result("logits")

@pytest.fixture(scope="module")
def greedy_context_attention(greedy_state, decoder_params):
    if decoder_params["attention"] == "none":
        return None
    else:
        return greedy_state.get_result("context_attention")

@pytest.fixture(scope="module")
def greedy_log_probability(greedy_state):
    return greedy_state.get_result("log_probability")

@pytest.fixture(scope="module")
def greedy_output_log_probability(greedy_state):
    return greedy_state.get_result("output_log_probability").t()

@pytest.fixture(scope="module")
def greedy_outputs(greedy_state):
    output = greedy_state.get_result("outputs").t()
    return output

def test_rnn_outputs(greedy_rnn_outputs, forward_rnn_outputs, tensor_equal):
    assert tensor_equal(greedy_rnn_outputs, forward_rnn_outputs)

def test_context_attention(greedy_context_attention, forward_context_attention,
                           tensor_equal):
    if greedy_context_attention is None and forward_context_attention is None:
        return 
    else:
        assert tensor_equal(greedy_context_attention, 
                            forward_context_attention)

def test_logits(greedy_logits, forward_logits, tensor_equal):
    assert tensor_equal(greedy_logits, forward_logits)

def test_log_probability(greedy_log_probability, forward_log_probability,
                         tensor_equal):
    assert tensor_equal(greedy_log_probability, forward_log_probability)

def test_outputs(greedy_outputs, forward_outputs, tensor_equal):
    assert tensor_equal(greedy_outputs, forward_outputs)


def test_output_log_probability(forward_output_log_probability, 
                                greedy_output_log_probability, tensor_equal):
    assert tensor_equal(forward_output_log_probability, 
                        greedy_output_log_probability)

@pytest.mark.xfail
def test_context_mask():
    pass

@pytest.mark.xfail
def test_resort():
    pass

@pytest.mark.xfail
def test_early_stop_no_collect():
    pass

@pytest.mark.xfail
def test_early_stop_do_collect():
    pass

@pytest.mark.xfail
def test_backprop():
    pass
