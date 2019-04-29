import pytest
import torch
import nnsum2.torch as ntorch
from nnsum2.seq2seq.decoders import RNN

@pytest.fixture(scope="function",
                params=["none", "bilinear", "accum_bilinear"],
                ids=["no_an", "blin_an", "acc_blin"])
def attention_mechanism(request, attention_mechanism_builder):
    return attention_mechanism_builder(request.param)

@pytest.fixture(scope="function")
def model(input_embedding_context, rnn, attention_mechanism,
          output_embedding_context):
    decoder = RNN(
        input_embedding_context=input_embedding_context,
        rnn=rnn,
        context_attention=attention_mechanism,
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
                if "source_mask" in context else None
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
        ("target_logits", 1), 
        ("rnn_input", 1),
        ("rnn_output", 1),
        ("decoder_state", 1),
        ("context_attention", 1),
        ("attention_output", 1),
        ("context_attention_state", 0),
        ("output", 0),
        ("log_probability", 1),
    ],
    ids=[
        "tgt_logits", "rnn_in", "rnn_out", "dec_state", "ctx_attn", "ctx_out",
        "ctx_attn_state", "output", "log_prob"
    ],
)
def test_sgl_eq_batch_fwd(batch_inputs_factory, context_factory, 
                          model, field_info, 
                          rnn_state=None, 
                          attention_state=None, mask_source=True):

    batch_inputs = batch_inputs_factory('simple')                          
    batch_context = context_factory(mask_source)
    s_is, s_cs, s_rs, s_as = convert_batch_inputs_to_singleton_inputs(
        batch_inputs, rnn_state, batch_context, attention_state)

    batch_fs = model(rnn_state, batch_inputs["target_input_features"],
                     batch_context, attention_state, compute_output=True,
                     compute_log_probability=True)
    batch_result = batch_fs[field_info[0]]
    
    single_fs = []
    for s_input, s_context, s_rnn_state, s_attn_state in zip(
            s_is, s_cs, s_rs, s_as): 
        single_fs.append(
            model(s_rnn_state, s_input["target_input_features"], 
                  s_context, s_attn_state, compute_output=True,
                  compute_log_probability=True))

    if batch_fs[field_info[0]] is None:
        assert all([fs[field_info[0]] is None for fs in single_fs])
    else:
        single_result = ntorch.cat(
            [fs[field_info[0]] for fs in single_fs], dim=field_info[1])
        assert ntorch.allclose(single_result, batch_result, rtol=0, atol=1e-5)
