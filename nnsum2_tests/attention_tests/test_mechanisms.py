import pytest
import torch
import nnsum2
from nnsum2.attention import BiLinearMechanism, FeedForwardMechanism

# TODO make this a factory method that adapts to different context sizes
MECHANISM_LIST = [
    {
        "name": "bilinear", 
        "mechanism": BiLinearMechanism(),
        "accumulate": False,
    },
    {
        "name": "acc_bilinear", 
        "mechanism": BiLinearMechanism(accumulate=True),
        "accumulate": True,
    },
    {
        "name": "feedforward", 
        "mechanism": FeedForwardMechanism(
            hidden_dims=8, key_dims=16, query_dims=16),
        "accumulate": False,
    },
    {
        "name": "acc_feedforward", 
        "mechanism": FeedForwardMechanism(
            accumulate=True,
            learn_accumulator_weights=False,
            hidden_dims=8, key_dims=16, query_dims=16),
        "accumulate": True,
    },
    {
        "name": "param_acc_feedforward", 
        "mechanism": FeedForwardMechanism(
            accumulate=True,
            learn_accumulator_weights=True,
            hidden_dims=8, key_dims=16, query_dims=16),
        "accumulate": True,
    },


]



@pytest.mark.parametrize("mechanism", MECHANISM_LIST, ids=lambda x: x["name"])
def test_sequential_matches_batch(mechanism, key, query, key_mask, batch_size,
                                  key_length, query_length,
                                  init_accumulator):

    reward = torch.FloatTensor(query_length, batch_size, key_length).normal_()
    if init_accumulator is not None:
        init_state = init_accumulator["accumulator"]
    else:
        init_state = None

    # Run batch mode.
    batch_attention, batch_state, batch_values = mechanism["mechanism"](
        key, query, key_mask=key_mask, state=init_state)

    if not mechanism["accumulate"]:
        assert batch_state is None
    (
        (batch_attention * reward).mean() + ((batch_values - 0)**2).mean()
    ).backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)
    if mechanism["accumulate"] and init_accumulator is not None:
        batch_accum_grad = init_state.grad.clone()
        init_state.grad.fill_(0)

    # Run sequentially and collect back into batch.
    query_steps = query.split(1, dim=0)

    step_state = init_state
    step_attentions = []
    step_values = []
    for query_step in query_steps:
        step_attention, step_state, step_value = mechanism["mechanism"](
            key, query_step, key_mask=key_mask, state=step_state)
        if not mechanism["accumulate"]:
            assert step_state is None
        step_attentions.append(step_attention)
        step_values.append(step_value)
    
    step_values = torch.cat(step_values, dim=0)
    step_attentions = torch.cat(step_attentions, dim=0)
    (
        (step_attentions * reward).mean() + ((step_values - 0)**2).mean()
    ).backward()
    step_key_grad = key.grad.clone()
    step_query_grad = query.grad.clone()
    if mechanism["accumulate"] and init_accumulator is not None:
        step_accum_grad = init_state.grad.clone()
        init_state.grad.fill_(0)

    assert torch.allclose(step_key_grad, batch_key_grad)
    assert torch.allclose(step_query_grad, batch_query_grad)
    assert torch.allclose(step_attentions, batch_attention)
    assert torch.allclose(step_values, batch_values, atol=1e-5, rtol=0)
    if mechanism["accumulate"] and init_accumulator is not None:
        assert torch.allclose(step_accum_grad, batch_accum_grad)


@pytest.mark.parametrize("mechanism", MECHANISM_LIST, ids=lambda x: x["name"])
def test_singleton_matches_batch(mechanism, key, query, key_mask, batch_size,
                                 key_length, query_length,
                                 init_accumulator):
    reward = torch.FloatTensor(query_length, batch_size, key_length).normal_()
    if init_accumulator is not None:
        init_state = init_accumulator["accumulator"]
    else:
        init_state = None

    # Run batch mode.
    batch_attention, batch_state, batch_values = mechanism["mechanism"](
        key, query, key_mask=key_mask, state=init_state)
    if not mechanism["accumulate"]:
        assert batch_state is None
    (
        (batch_attention * reward).mean() + ((batch_values - 0)**2).mean()
    ).backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)
    if mechanism["accumulate"] and init_accumulator is not None:
        batch_accum_grad = init_state.grad.clone()
        init_state.grad.fill_(0)

    # Run singleton and collect back into batch.
    single_keys = key.split(1, dim=0)
    single_queries = query.split(1, dim=1)
    if key_mask is not None:
        single_key_masks = key_mask.split(1, dim=0)
    else:
        single_key_masks = [None] * len(single_keys)
    if init_accumulator is not None:
        single_init_states = init_state.split(1, dim=0)
    else:
        single_init_states = [None] * len(single_keys)

#    step_state = init_state
    single_attentions = []
    single_values = []
    single_states = []
    for single_key, single_query, single_key_mask, single_init_state in zip(
            single_keys, single_queries, single_key_masks, single_init_states):
        single_attention, single_state, single_value = mechanism["mechanism"](
            single_key, single_query, key_mask=single_key_mask,
            state=single_init_state)
        if not mechanism["accumulate"]:
            assert single_state is None
        else:
            single_states.append(single_state)
        single_attentions.append(single_attention)
        single_values.append(single_value)

    if mechanism["accumulate"]:
        single_states = torch.cat(single_states, dim=0)
    single_values = torch.cat(single_values, dim=1)
    single_attentions = torch.cat(single_attentions, dim=1)
    (
        (single_attentions * reward).mean() + ((single_values - 0)**2).mean()
    ).backward()
    single_key_grad = key.grad.clone()
    single_query_grad = query.grad.clone()
    if mechanism["accumulate"] and init_accumulator is not None:
        single_accum_grad = init_state.grad.clone()
        init_state.grad.fill_(0)

    assert torch.allclose(single_attentions, batch_attention)
    assert torch.allclose(single_values, batch_values, rtol=0, atol=1e-5)

    if mechanism["accumulate"]:
        assert torch.allclose(single_states, batch_state)

    assert torch.allclose(single_key_grad, batch_key_grad)
    assert torch.allclose(single_query_grad, batch_query_grad, 
                          rtol=0, atol=1e-5)
    if mechanism["accumulate"] and init_accumulator is not None:
        assert torch.allclose(single_accum_grad, batch_accum_grad)

@pytest.mark.parametrize("mechanism", MECHANISM_LIST, ids=lambda x: x["name"])
@pytest.mark.parametrize("loss_src", ["attention", "value"],
                         ids=["loss_src=attn", "loss_src=value"]) 
def test_key_masking_gradient(mechanism, key, query, key_mask, 
                              init_accumulator, loss_src):
    if init_accumulator:
        init_state = init_accumulator["accumulator"]
    else:
        init_state = None
    attention, state, value = mechanism["mechanism"](
        key, query, key_mask=key_mask, state=init_state)

    if loss_src == "attention":
        reward = torch.FloatTensor(attention.size()).normal_()
        (attention * reward).mean().backward()
    elif loss_src == "value":
        ((value - 0)**2).mean().backward()
    else:
        raise Exception("loss_src must be either 'attention' or 'value'.")

    if key_mask is None:
        assert torch.all(key.grad.ne(0))
        if mechanism["accumulate"] and init_accumulator is not None: 
            accum_grad = init_state.grad
            assert torch.all(accum_grad.ne(0))
    else:
        key_mask_1 = key_mask.unsqueeze(2) 
        assert torch.all(key.grad.masked_select(key_mask_1).eq(0))
        assert torch.all(key.grad.masked_select(~key_mask_1).ne(0))

        if mechanism["accumulate"] and init_accumulator is not None: 
            accum_grad = init_state.grad
            key_mask_1 = key_mask.unsqueeze(1) 
            assert torch.all(accum_grad.masked_select(key_mask_1).eq(0))
            assert torch.all(accum_grad.masked_select(~key_mask_1).ne(0))

    assert torch.all(query.grad.ne(0))

@pytest.mark.parametrize("mechanism", MECHANISM_LIST, ids=lambda x: x["name"])
def test_masking_output(mechanism, key, query, key_mask):
    attention, state, value = mechanism["mechanism"](
        key, query, key_mask=key_mask)
    if mechanism["accumulate"]:
        assert state is not None
    else:
        assert state is None

    if key_mask is None:
        assert torch.all(attention.ne(0))
    else:
        key_mask_1 = key_mask.unsqueeze(0) 
        assert torch.all(attention.masked_select(key_mask_1).eq(0))
        assert torch.all(attention.masked_select(~key_mask_1).ne(0))

    assert torch.allclose(
        attention.sum(2), 
        torch.FloatTensor(attention.size()[:2]).fill_(1.))

@pytest.mark.parametrize("mechanism", MECHANISM_LIST, ids=lambda x: x["name"])
def test_composition(mechanism, key, query, key_mask, init_accumulator,
                     hidden_size, batch_size, query_length, key_length):
    if init_accumulator:
        init_state = init_accumulator["accumulator"]
    else:
        init_state = None

    attention, state, value = mechanism["mechanism"](
        key, query, key_mask=key_mask, state=init_state)

    ref_value = torch.FloatTensor(query_length, batch_size, hidden_size)
    ref_value.fill_(0)
    for q in range(query_length):
        for b in range(batch_size):
            for h in range(hidden_size):
                for k in range(key_length):
                    ref_value[q,b,h] += attention[q,b,k] * key[b,k,h]

    assert torch.allclose(ref_value, value, rtol=0, atol=1e-5)
