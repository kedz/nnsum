import pytest
import torch
import nnsum2.torch as ntorch
from nnsum2.seq2seq.searches import GreedySearch, BeamSearch, SearchState


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

def mask_forward_state(state, field, mask):
    result = state[field]
    if result is None:
        return result
    dim_names = state._dim_names[field]
    batch_index = dim_names.index("batch")
    seq_index = dim_names.index("sequence")
    mask_dims = [1] * len(dim_names)
    mask_dims[batch_index] = result.size(batch_index)
    mask_dims[seq_index] = result.size(seq_index)

    if batch_index < seq_index:
        return result.masked_fill(mask.view(*mask_dims), 0)
    else:
        return result.masked_fill(mask.t().contiguous().view(*mask_dims), 0)

RNN_ATTN_TYPES = ["none", "bilinear", "accum_bilinear"]
RNN_PG_ATTN_TYPES = ["bilinear", "accum_bilinear"]

RNN_RESULT_TYPES = [
    {"field": "rnn_input"},
    {"field": "rnn_output"},
    {"field": "context_attention"},
    {"field": "attention_output"},
    {"field": "target_logits"},
    {"field": "log_probability"},
    {"field": "context_attention_state", "step_forward": True},
    {"field": "decoder_state", "step_forward": True},
]

RNN_PG_RESULT_TYPES = [
    {"field": "rnn_input"},
    {"field": "rnn_output"},
    {"field": "context_attention"},
    {"field": "copy_switch"},
    {"field": "attention_output"},
    {"field": "context_attention_state", "step_forward": True},
    {"field": "decoder_state", "step_forward": True},
    {"field": "generator_logits"},
    {"field": "generator_distribution"},
    {"field": "pointer_distribution"},
    {"field": "generator_probability"},
    {"field": "pointer_probability"},
    {"field": "log_probability"},
]

def param_generator(dec_result_types):
    decoder_types = ["rnn", "rnn_pointer_generator"]
    dec_attn_types = {
        "rnn": RNN_ATTN_TYPES,
        "rnn_pointer_generator": RNN_PG_ATTN_TYPES,
    }
   
    for dec_type in decoder_types:
        for attn in dec_attn_types[dec_type]:
            for result in dec_result_types[dec_type]:
                p = {
                    "decoder_params": {
                        "type": dec_type,
                        "attention_mechanism": attn,
                    },
                    "result_params": dict(result),
                }
                if p["decoder_params"]["attention_mechanism"] == "none" \
                        and "attention" in p["result_params"]["field"]:
                    p["result_params"]["expect_none"] = True

                if "accum" not in p["decoder_params"]["attention_mechanism"] \
                        and "attention_state" in p["result_params"]["field"]:
                    p["result_params"]["expect_none"] = True
                yield p

def step_forward(decoder, state, context, inputs):
    states = []
    for step_input in inputs.split(1, 1):
        state = decoder(state["decoder_state"], step_input, context, 
            state.get("context_attention_state", None))
        states.append(state)
    return SearchState.consolidate(states)

def param_name(params):
    return params["decoder_params"]["type"] + "-" + \
        params["decoder_params"]["attention_mechanism"] + "-" + \
        params["result_params"]["field"]

@pytest.mark.parametrize("test_params", 
    param_generator(
        {
            "rnn": RNN_RESULT_TYPES,
            "rnn_pointer_generator": RNN_PG_RESULT_TYPES,
        }
    ),
    ids=param_name)
def test_consistent_with_forward(test_params, decoder_factory, 
                                 init_rnn_state, context,
                                 target_vocab):

    result_field = test_params["result_params"]["field"]
    expect_none = test_params["result_params"].get("expect_none", False)

    decoder = decoder_factory(**test_params["decoder_params"])

    search = BeamSearch(decoder, {"encoder_state": init_rnn_state},
                        context, max_steps=10, return_incomplete=True,
                        beam_size=4)
    beam_output = search.get_result("output", mask=True)
    beam_mask = search._selector_mask
    print(beam_output)
    print(beam_output.size())
    input()

    fwd_inputs = output2inputs(beam_output, target_vocab)
    fwd_init_state = search._initialize_search_state(
        {"encoder_state": init_rnn_state})
    fwd_context = search._initialize_context(context)

    if test_params["result_params"].get("step_forward", False):
        fwd_state = step_forward(
            decoder, fwd_init_state, fwd_context,
            fwd_inputs)
    else:
        fwd_state = decoder(
            fwd_init_state["decoder_state"], fwd_inputs, fwd_context, 
            fwd_init_state.get("context_attention_state", None),
            compute_log_probability=True)

    fwd_result = mask_forward_state(fwd_state, result_field, beam_mask)
    beam_result = search.get_result(result_field, mask=True)
    
    if expect_none:
        assert fwd_result is None
        assert beam_result is None

    else:
        assert ntorch.allclose(fwd_result, beam_result, rtol=0, atol=1e-5)

@pytest.mark.parametrize("test_params", 
    param_generator(
        {
            "rnn": RNN_RESULT_TYPES + [
                {"field": "output"}, 
                {"field": "cumulative_log_probability"}
            ],
            "rnn_pointer_generator": RNN_PG_RESULT_TYPES + [
                {"field": "output"},
                {"field": "cumulative_log_probability"}
            ],
        }
    ),
    ids=param_name)
def test_greedy_eq_beam1(test_params, decoder_factory, 
                         init_rnn_state, context,
                         target_vocab):

    result_field = test_params["result_params"]["field"]
    expect_none = test_params["result_params"].get("expect_none", False)

    decoder = decoder_factory(**test_params["decoder_params"])

    beam_search = BeamSearch(decoder, {"encoder_state": init_rnn_state},
                             context, max_steps=10, return_incomplete=True,
                             beam_size=1)
    beam_result = beam_search.get_result(result_field, mask=True)
    
    greedy_search = GreedySearch(decoder, {"encoder_state": init_rnn_state},
                                 context, max_steps=10, return_incomplete=True,
                                 compute_cumulative_log_probability=True)
    greedy_result = greedy_search.get_result(result_field, mask=True)

    if expect_none:
        assert greedy_result is None
        assert beam_result is None
    else:
        ntorch.allclose(greedy_result, beam_result)


@pytest.mark.parametrize("test_params", 
    param_generator(
        {
            "rnn": [{"field": "_"}],
            "rnn_pointer_generator": [{"field": "_"}],
        }
    ),
    ids=param_name)
def test_score_is_avg_logprob(test_params, decoder_factory, 
                              init_rnn_state, context,
                              target_vocab):

    decoder = decoder_factory(**test_params["decoder_params"])

    beam_search = BeamSearch(decoder, {"encoder_state": init_rnn_state},
                             context, max_steps=10, return_incomplete=True,
                             beam_size=4)
    beam_clp = beam_search.get_result(
        "cumulative_log_probability", mask=True).squeeze(2)
  
    lengths = beam_search._lengths 
    beam_clp = beam_clp.t().contiguous().view(3, 4, -1)
    lengths = lengths.unsqueeze(2)
    indexer = lengths - 1 
    expected_score = (beam_clp.gather(2, indexer) / lengths.float()) \
        .squeeze(2)

    assert ntorch.allclose(expected_score, beam_search.beam_scores)

@pytest.mark.parametrize("test_params", 
    param_generator(
        {
            "rnn": [{"field": "_"}],
            "rnn_pointer_generator": [{"field": "_"}],
        }
    ),
    ids=param_name)
def test_sorted_scores(test_params, decoder_factory, 
                       init_rnn_state, context,
                       target_vocab):

    decoder = decoder_factory(**test_params["decoder_params"])

    beam_search = BeamSearch(decoder, {"encoder_state": init_rnn_state},
                             context, max_steps=10, return_incomplete=True,
                             beam_size=4)
    beam_search.search()
    expected = torch.arange(4).view(1, -1).repeat(3, 1)
    
    assert torch.all(
        expected == \
            torch.sort(beam_search.beam_scores, dim=1, descending=True)[1]
    )

    assert torch.all(
        beam_search.beam_scores == \
            torch.sort(beam_search.beam_scores, dim=1, descending=True)[0]
    )
