import pytest
import torch
import numpy as np

import nnsum2
import nnsum2.torch as ntorch
from nnsum2.seq2seq.decoders import RNNPointerGenerator


@pytest.fixture(scope="function",
                params=["bilinear", "accum_bilinear"],
                ids=["blin_an", "acc_blin"])
def attention_mechanism(request, attention_mechanism_builder):
    return attention_mechanism_builder(request.param)

@pytest.fixture(scope="function")
def model(input_embedding_context, rnn, attention_mechanism,
          output_embedding_context, context_dims):

    decoder = RNNPointerGenerator(
        input_embedding_context=input_embedding_context,
        rnn=rnn,
        context_attention=attention_mechanism,
        copy_switch=nnsum2.layers.FullyConnected(
            in_feats=2 * context_dims + rnn.output_dims,
            out_feats=1,
            dropout=0.0, 
            activation="Sigmoid",
        ),
        output_embedding_context=output_embedding_context,
    )
    decoder.eval()
    return decoder

def convert_batch_inputs_to_singleton_inputs(inputs, rnn_state, context, 
                                             attention_state):
    singleton_inputs = []
    singleton_context = []
    singleton_rnn_states = []
    singleton_attention_states = []
    for b in range(3):
        singleton_inputs.append({
            "target_input_features": {
                k: v[b:b+1]
                for k, v in inputs["target_input_features"].items()
            },
            "target_mask": inputs["target_mask"][b:b+1],
            "target_lengths": inputs["target_lengths"][b:b+1]
        })
        singleton_context.append({
            "encoder_output": context["encoder_output"][b:b+1],
            "source_mask": context["source_mask"][b:b+1] \
                if "source_mask" in context else None,
            "extended_vocab": context["extended_vocab"],
            "source_extended_vocab_map": \
                context["source_extended_vocab_map"][b:b+1]
        })

    if rnn_state is None:
        singleton_rnn_states.extend([None] * 3)
    else:
        raise Exception("make single rnn state")
    if attention_state is None:
        singleton_attention_states.extend([None] * 3)
    else:
        raise Exception("make singleatnn state")

    return singleton_inputs, singleton_context, singleton_rnn_states, singleton_attention_states

@pytest.mark.parametrize(
    "field_info", 
    [
        ("rnn_input", 1),
        ("rnn_output", 1),
        ("decoder_state", 1),
        ("context_attention", 1),
        ("attention_output", 1),
        ("context_attention_state", 0),
        ("copy_switch", 1),
        ("generator_distribution", 1), 
        ("pointer_distribution", 1), 
        ("generator_probability", 1),
        ("pointer_probability", 1),
        ("output", 0),
        ("log_probability", 1),
    ],
    ids=[
        "rnn_in", "rnn_out", "rnn_state", "ctx_attn", "ctx_out",
        "ctx_attn_state", "switch", "gen_dist", "ptr_dist", 
        "gen_prob", "ptr_prob", "output", "log_prob",
    ],
)
def test_sgl_eq_batch_fwd(batch_inputs_factory, context_factory, 
                          model, field_info, 
                          rnn_state=None, 
                          attention_state=None, mask_source=True):
    batch_inputs = batch_inputs_factory("pg")           
    batch_context = context_factory(mask_source, is_pg=True)
    s_is, s_cs, s_rs, s_as = convert_batch_inputs_to_singleton_inputs(
        batch_inputs, rnn_state, batch_context, attention_state)

    batch_fs = model(rnn_state, batch_inputs["target_input_features"],
                     batch_context, attention_state, compute_output=True)
    batch_result = batch_fs[field_info[0]]
    
    single_fs = []
    for s_input, s_context, s_rnn_state, s_attn_state in zip(
            s_is, s_cs, s_rs, s_as): 
        single_fs.append(
            model(s_rnn_state, s_input["target_input_features"], 
                  s_context, s_attn_state, compute_output=True))

    if batch_fs[field_info[0]] is None:
        assert all([fs[field_info[0]] is None for fs in single_fs])
    else:
        if field_info[0] == "log_probability":
            pad_value = np.log(1e-12)
        else:
            pad_value = 0
        single_result = ntorch.pad_and_cat(
            [fs[field_info[0]] for fs in single_fs], dim=field_info[1],
            pad_value=pad_value)
        assert ntorch.allclose(single_result, batch_result, rtol=0, atol=1e-5)

# This could be simpler with a hand made source and expected projection.
def test_copy_projection(model, context_factory):

    context = context_factory(True, is_pg=True)
    vocab = context["extended_vocab"]
    source = context["source_extended_vocab_map"]
    steps = 3
    attn_logit = source.new().float().new(
        *([steps] + list(source.size()))).normal_().masked_fill(
        context["source_mask"].unsqueeze(0), float("-inf"))
    attn = torch.softmax(attn_logit, dim=2)

    expected_proj_attn = attn.new(steps, source.size(0), len(vocab)).fill_(0)
    for step in range(steps):
        for b in range(source.size(0)):
            for i in range(source.size(1)):   
                a = attn[step, b,i]
                idx = source[b, i]
                expected_proj_attn[step, b, idx] = a
               
    actual_proj_attn = model._map_attention_to_vocab(attn, source, len(vocab))
    assert ntorch.allclose(actual_proj_attn, expected_proj_attn)
