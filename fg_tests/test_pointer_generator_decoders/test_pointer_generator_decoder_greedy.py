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
def greedy_state(model, train_data):
    encoder_state, context = model.get_context(train_data)
    return s2s.GreedySearch(model.decoder, encoder_state, context, 
                            max_steps=10)

@pytest.fixture(scope="module")
def eval_data(greedy_state, target_vocab, train_data):

    tokens = greedy_state.get_result("output").t()
    inputs = tokens.new(tokens.size()).fill_(target_vocab.pad_index)
    inputs[:,0].fill_(target_vocab.start_index)
    copy_targets = tokens.new(tokens.size()).fill_(target_vocab.pad_index)
    lengths = tokens.new(tokens.size(0))
    
    for i, row in enumerate(tokens):
        try:
            length = row.tolist().index(target_vocab.stop_index) + 1
            inputs[i,1:length].copy_(row[:length-1])
            copy_targets[i,:length].copy_(row[:length])
            lengths[i] = length
        except ValueError:
            lengths[i] = 0
            continue
    inputs = inputs.masked_fill(inputs.ge(len(target_vocab)),
                                target_vocab.unknown_index)
    outputs = copy_targets.masked_fill(copy_targets.ge(len(target_vocab)),
                                       target_vocab.unknown_index)

    eval_data = {
        "source_lengths": train_data["source_lengths"],    
        "source_input_features": train_data["source_input_features"],
        "source_mask": train_data["source_mask"],
        "source_vocab_map": train_data["source_vocab_map"],
        "target_input_features": {"tokens": inputs},
        "target_output_features": {"tokens": outputs},
        "target_lengths": lengths}
    return eval_data

@pytest.fixture(scope="module")
def forward_state(model, eval_data):
    return model.get_state(eval_data, compute_output=True)

@pytest.fixture(scope="module")
def eval_mask(target_vocab, eval_data):
    mask = eval_data["target_input_features"]["tokens"].eq(
        target_vocab.pad_index)
    return mask

def get_forward_field(field, forward_state, eval_mask):
    output = forward_state[field]
    steps = output.size(0)
    batch_size = output.size(1)
    mask = eval_mask.t()
    while mask.dim() < output.dim():
        mask = mask.unsqueeze(-1)
    output = output.masked_fill(mask, 0.)
    return output

@pytest.fixture(scope="module")
def forward_rnn_outputs(forward_state, eval_mask):
    return get_forward_field("rnn_output", forward_state, eval_mask)

@pytest.fixture(scope="module")
def forward_context_attention(forward_state, eval_mask):
    return get_forward_field("context_attention", forward_state, eval_mask)

@pytest.fixture(scope="module")
def forward_logits(forward_state, eval_mask):
    return get_forward_field("target_logits", forward_state, eval_mask)

@pytest.fixture(scope="module")
def forward_log_probability(forward_state, eval_mask):
    return get_forward_field("log_probability", forward_state, eval_mask)

@pytest.fixture(scope="module")
def forward_output(forward_state, eval_mask):
    return get_forward_field("output", forward_state, eval_mask)

@pytest.fixture(scope="module")
def forward_output_log_probability(forward_state, eval_mask):    
    return get_forward_field("output_log_probability", 
                             forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_rnn_outputs(greedy_state):
    return greedy_state.get_result("rnn_output")

@pytest.fixture(scope="module")
def greedy_logits(greedy_state):
    return greedy_state.get_result("target_logits")

@pytest.fixture(scope="module")
def greedy_context_attention(greedy_state):
    return greedy_state.get_result("context_attention")

@pytest.fixture(scope="module")
def greedy_log_probability(greedy_state):
    return greedy_state.get_result("log_probability")

@pytest.fixture(scope="module")
def greedy_output_log_probability(greedy_state):
    return greedy_state.get_result("output_log_probability")

@pytest.fixture(scope="module")
def greedy_output(greedy_state):
    output = greedy_state.get_result("output")
    return output

def test_rnn_outputs(greedy_rnn_outputs, forward_rnn_outputs, tensor_equal):
    assert tensor_equal(greedy_rnn_outputs, forward_rnn_outputs)

def test_context_attention(greedy_context_attention, forward_context_attention,
                           tensor_equal):
    assert tensor_equal(greedy_context_attention, forward_context_attention)

def test_logits(greedy_logits, forward_logits, tensor_equal):
    assert tensor_equal(greedy_logits, forward_logits)

def test_log_probability(greedy_log_probability, forward_log_probability,
                         tensor_equal):
    assert tensor_equal(greedy_log_probability, forward_log_probability)

def test_output(greedy_output, forward_output, tensor_equal):
    print()
    print(greedy_output)
    print(forward_output)
    assert tensor_equal(greedy_output, forward_output)

def test_output_log_probability(forward_output_log_probability, 
                                greedy_output_log_probability, tensor_equal):
    assert tensor_equal(forward_output_log_probability, 
                        greedy_output_log_probability)

@pytest.fixture(scope="module")
def forward_copy_attention(forward_state, eval_mask):
    return get_forward_field("copy_attention", forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_copy_attention(greedy_state):
    return greedy_state.get_result("copy_attention")

def test_copy_attention(greedy_copy_attention, forward_copy_attention,
                        tensor_equal):
    assert tensor_equal(forward_copy_attention, greedy_copy_attention)

@pytest.fixture(scope="module")
def forward_source_probability(forward_state, eval_mask):
    return get_forward_field("source_probability", forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_source_probability(greedy_state):
    return greedy_state.get_result("source_probability")

def test_source_probability(greedy_source_probability, 
                            forward_source_probability,
                            tensor_equal):
    assert tensor_equal(forward_source_probability, greedy_source_probability)

@pytest.fixture(scope="module")
def forward_pointer_probability(forward_state, eval_mask):
    return get_forward_field("pointer_probability", forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_pointer_probability(greedy_state):
    return greedy_state.get_result("pointer_probability")

def test_pointer_probability(greedy_pointer_probability, 
                            forward_pointer_probability,
                            tensor_equal):
    assert tensor_equal(forward_pointer_probability, 
                        greedy_pointer_probability)

@pytest.fixture(scope="module")
def forward_target_probability(forward_state, eval_mask):
    return get_forward_field("target_probability", forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_target_probability(greedy_state):
    return greedy_state.get_result("target_probability")

def test_target_probability(greedy_target_probability, 
                            forward_target_probability,
                            tensor_equal):
    assert tensor_equal(forward_target_probability, 
                        greedy_target_probability)

@pytest.fixture(scope="module")
def forward_generator_probability(forward_state, eval_mask):
    return get_forward_field("generator_probability", forward_state, eval_mask)

@pytest.fixture(scope="module")
def greedy_generator_probability(greedy_state):
    return greedy_state.get_result("generator_probability")

def test_generator_probability(greedy_generator_probability, 
                            forward_generator_probability,
                            tensor_equal):
    assert tensor_equal(forward_generator_probability, 
                        greedy_generator_probability)

@pytest.mark.xfail
def test_early_stop_no_collect():
    pass

@pytest.mark.xfail
def test_early_stop_do_collect():
    pass

@pytest.mark.xfail
def test_context_mask():
    pass

@pytest.mark.xfail
def test_backprop():
    pass
