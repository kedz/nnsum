import pytest
import torch
import nnsum.seq2seq as s2s


def data_param_gen():
    return [{"target_texts": "beam"}]

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
def beam_state(decoder, encoder_state, context):
    return s2s.BeamSearch(decoder, encoder_state, context, max_steps=10, 
                          beam_size=3, sort_by_score=False)

@pytest.fixture(scope="module")
def eval_data(beam_state, vocab):
    tokens = beam_state.get_result("output").permute(1, 2, 0).view(6,-1)

    new_inputs = tokens.new(6, tokens.size(1)).fill_(vocab.pad_index)
    new_inputs[:,0].fill_(vocab.start_index)
    new_outputs = tokens.new(6, tokens.size(1)).fill_(vocab.pad_index)
    new_lengths = tokens.new(6).long()
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
def eval_mask(vocab, eval_data, batch_size, beam_size):
    mask = eval_data["target_input_features"]["tokens"].eq(vocab.pad_index)
    mask = mask.view(batch_size, beam_size, -1)
    return mask

@pytest.fixture(scope="module")
def forward_state(decoder, eval_data, encoder_state, context, 
                  initialize_decoder):
    encoder_state, context = initialize_decoder(encoder_state, context)    
    fwd_state = decoder(
        encoder_state,
        eval_data["target_input_features"],
        context)
   
    fwd_state["log_probability"] = torch.log_softmax(
        fwd_state["target_logits"], dim=2)

    return fwd_state

def get_forward_field(field, forward_state, batch_size, beam_size, eval_mask):
    output = forward_state[field]
    if output is None:
        return None
    steps = output.size(0)
    output = output.view(steps, batch_size, beam_size, -1).masked_fill(
        eval_mask.permute(2, 0, 1).unsqueeze(-1), 0.)
    return output

@pytest.fixture(scope="module")
def forward_rnn_outputs(forward_state, batch_size, beam_size, eval_mask):
    return get_forward_field("rnn_output", forward_state, batch_size,
                             beam_size, eval_mask)

@pytest.fixture(scope="module")
def forward_context_attention(forward_state, batch_size, beam_size, eval_mask):
    if "context_attention" not in forward_state:
        return None
    else:
        return get_forward_field("context_attention", forward_state,
                                 batch_size, beam_size, eval_mask)

@pytest.fixture(scope="module")
def forward_logits(forward_state, batch_size, beam_size, eval_mask):
    return get_forward_field("target_logits", forward_state, batch_size, beam_size, 
                             eval_mask)

@pytest.fixture(scope="module")
def forward_log_probability(forward_state, batch_size, beam_size, eval_mask):
    return get_forward_field("log_probability", forward_state, batch_size,
                             beam_size, eval_mask)

@pytest.fixture(scope="module")
def forward_output_log_probability(forward_log_probability, eval_data,
                                   batch_size, beam_size):    
    tgts = eval_data["target_output_features"]["tokens"].t()
    steps = tgts.size(0)
    tgts = tgts.view(steps, batch_size, beam_size, 1)
    return forward_log_probability.gather(3, tgts).squeeze(-1)

#@pytest.fixture(scope="module")
#def forward_outputs(forward_state, batch_size, beam_size, eval_mask):
#    print(forward_state["log_probability"].size())
#    output = forward_state["log_probability"].max(2)[1].t()
#    steps = output.size(1)
#    output = output.view(batch_size, beam_size, steps).masked_fill(
#        eval_mask, 0)
#    return output

@pytest.fixture(scope="module")
def beam_rnn_outputs(beam_state):
    return beam_state.get_result("rnn_output")

@pytest.fixture(scope="module")
def beam_logits(beam_state):
    return beam_state.get_result("target_logits")

@pytest.fixture(scope="module")
def beam_context_attention(beam_state, decoder_params):
    if decoder_params["attention"] == "none":
        return None
    else:
        return beam_state.get_result("context_attention")

@pytest.fixture(scope="module")
def beam_log_probability(beam_state):
    return beam_state.get_result("log_probability")

@pytest.fixture(scope="module")
def beam_output_log_probability(beam_state):
    return beam_state.get_result("output_log_probability")

#@pytest.fixture(scope="module")
#def beam_outputs(beam_state):
#    output = beam_state.get_result("outputs").permute(1,2,0)
#    return output

def test_rnn_outputs(beam_rnn_outputs, forward_rnn_outputs, tensor_equal):
    assert tensor_equal(beam_rnn_outputs, forward_rnn_outputs)

def test_context_attention(beam_context_attention, forward_context_attention,
                           tensor_equal):
    if beam_context_attention is None and forward_context_attention is None:
        return 
    else:
        assert tensor_equal(beam_context_attention, forward_context_attention)

def test_logits(beam_logits, forward_logits, tensor_equal):
    assert tensor_equal(beam_logits, forward_logits)

def test_log_probability(beam_log_probability, forward_log_probability,
                         tensor_equal):
    assert tensor_equal(beam_log_probability, forward_log_probability)

def test_output_log_probability(forward_output_log_probability, 
                                beam_output_log_probability, tensor_equal):
    assert tensor_equal(forward_output_log_probability, 
                        beam_output_log_probability)

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

@pytest.fixture(scope="module")
def backprop_config(decoder_params, decoder, encoder_state, context,
                    trainable_parameters, beam_size, check_gradient):
    optim = torch.optim.SGD(
        trainable_parameters(decoder, encoder_state, context), lr=5.0) 
    sch = s2s.BeamSearch(decoder, encoder_state, context, max_steps=10, 
                         beam_size=beam_size, sort_by_score=False)
    return {"optim": optim, "search": sch}

def test_rnn_outputs_backprop(backprop_config, named_trainable_parameters,
                              decoder, encoder_state, context, check_gradient):
    nz_grad_params = []
    z_grad_params = []
    for param in named_trainable_parameters(decoder, encoder_state, context):
        if param[0] == "context" or param[0].startswith("_predictor"):
            z_grad_params.append(param)
        else:
            nz_grad_params.append(param)
    check_gradient(backprop_config, "rnn_output", nz_grad_params,
                   zero_gradient_params=z_grad_params)

def test_context_attention_backprop(backprop_config, decoder_params,
                                    named_trainable_parameters,
                                    decoder, encoder_state, context,
                                    check_gradient):
    if decoder_params["attention"] == "none":
        return
    nz_grad_params = []
    z_grad_params = []
    for param in named_trainable_parameters(decoder, encoder_state, context):
        if param[0].startswith("_predictor"):
            z_grad_params.append(param)
        else:
            nz_grad_params.append(param)
    check_gradient(backprop_config, "context_attention", nz_grad_params,
                   zero_gradient_params=z_grad_params)

def test_logits_backprop(backprop_config, named_trainable_parameters, decoder, 
                         encoder_state, context, check_gradient):
    check_gradient(backprop_config, "target_logits", 
                   named_trainable_parameters(decoder, encoder_state, context))

def test_log_probability_backprop(backprop_config, named_trainable_parameters,
                                  decoder, encoder_state, context, 
                                  check_gradient):
    check_gradient(backprop_config, "log_probability", 
                   named_trainable_parameters(decoder, encoder_state, context))

def test_output_log_probability_backprop(backprop_config, 
                                         named_trainable_parameters,
                                         decoder, encoder_state, context,
                                         check_gradient):
    check_gradient(backprop_config, "output_log_probability",
                   named_trainable_parameters(decoder, encoder_state, context))

@pytest.mark.xfail
def test_greedy_matches_1beam():
    pass

@pytest.mark.xfail
def test_reinforce_backprop():
    pass
