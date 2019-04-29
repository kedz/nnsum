import pytest
import torch
import nnsum2


def test_is_stateless():
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    assert kernel.is_stateless == False

def test_sequential_matches_batch(key, query, key_mask, init_accumulator, 
                                  batch_size, key_length, query_length):
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    reward = torch.FloatTensor(batch_size, query_length, key_length).normal_()

    # Run batch mode.
    batch_attention, batch_state = kernel(key, query, key_mask=key_mask,
                                          state=init_accumulator)
    (batch_attention * reward).mean().backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)
    if init_accumulator is not None:
        batch_accum_grad = init_accumulator["accumulator"].grad.clone()
        init_accumulator["accumulator"].grad.fill_(0)

    # Run sequentially and collect back into batch.
    query_steps = query.split(1, dim=0)

    step_state = init_accumulator
    step_attentions = []
    for query_step in query_steps:
        step_attention, step_state = kernel(key, query_step, key_mask=key_mask,
                                            state=step_state)
        step_attentions.append(step_attention)

    step_attentions = torch.cat(step_attentions, dim=1)
    (step_attentions * reward).mean().backward()
    step_key_grad = key.grad.clone()
    step_query_grad = query.grad.clone()
    if init_accumulator is not None:
        step_accum_grad = init_accumulator["accumulator"].grad.clone()
        init_accumulator["accumulator"].grad.fill_(0)

    assert torch.allclose(step_attentions, batch_attention)
    assert torch.allclose(step_state["accumulator"],
                          batch_state["accumulator"])
    assert torch.allclose(step_key_grad, batch_key_grad)
    assert torch.allclose(step_query_grad, batch_query_grad)
    if init_accumulator is not None:
        assert torch.allclose(step_accum_grad, batch_accum_grad)

def test_singleton_matches_batch(key, query, key_mask, init_accumulator,
                                 batch_size, key_length, query_length):
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    reward = torch.FloatTensor(batch_size, query_length, key_length).normal_()

    # Run batch mode.
    batch_attention, batch_state = kernel(key, query, key_mask=key_mask,
                                          state=init_accumulator)
    (batch_attention * reward).mean().backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)
    if init_accumulator is not None:
        batch_accum_grad = init_accumulator["accumulator"].grad.clone()
        init_accumulator["accumulator"].grad.fill_(0)

    # Run singleton and collect back into batch.
    single_keys = key.split(1, dim=0)
    single_queries = query.split(1, dim=1)
    if key_mask is not None:
        single_key_masks = key_mask.split(1, dim=0)
    else:
        single_key_masks = [None] * len(single_keys)
    if init_accumulator is not None:
        single_init_states = [
            {"accumulator": s} 
            for s in init_accumulator["accumulator"].split(1, dim=0)
        ]
    else:
        single_init_states = [None] * len(single_keys)

    single_accum = []
    single_attentions = []
    for single_key, single_query, single_key_mask, single_init_state in zip(
            single_keys, single_queries, single_key_masks, single_init_states):
        single_attention, single_state = kernel(
            single_key, single_query, key_mask=single_key_mask,
            state=single_init_state)
        single_attentions.append(single_attention)
        single_accum.append(single_state["accumulator"])
    
    single_accum = torch.cat(single_accum, dim=0)
    single_attentions = torch.cat(single_attentions, dim=0)
    (single_attentions * reward).mean().backward()
    single_key_grad = key.grad.clone()
    single_query_grad = query.grad.clone()
    if init_accumulator is not None:
        single_accum_grad = init_accumulator["accumulator"].grad.clone()
        init_accumulator["accumulator"].grad.fill_(0)

    assert torch.allclose(single_attentions, batch_attention)
    assert torch.allclose(single_accum, batch_state["accumulator"])
    assert torch.allclose(single_key_grad, batch_key_grad)
    assert torch.allclose(single_query_grad, batch_query_grad)
    if init_accumulator is not None:
        assert torch.allclose(single_accum_grad, batch_accum_grad)

def test_key_masking_gradient(key, query, key_mask, init_accumulator):
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    batch_attention, batch_state = kernel(key, query, key_mask=key_mask,
                                          state=init_accumulator)

    reward = torch.FloatTensor(batch_attention.size()).normal_()
    (batch_attention * reward).mean().backward()

    if key_mask is None:
        assert torch.all(key.grad.ne(0))
        if init_accumulator is not None: 
            accum_grad = init_accumulator["accumulator"].grad
            assert torch.all(accum_grad.ne(0))

    else:
        key_mask_2 = key_mask.unsqueeze(2) 
        assert torch.all(key.grad.masked_select(key_mask_2).eq(0))
        assert torch.all(key.grad.masked_select(~key_mask_2).ne(0))

        if init_accumulator is not None: 
            accum_grad = init_accumulator["accumulator"].grad
            key_mask_1 = key_mask.unsqueeze(1) 
            assert torch.all(accum_grad.masked_select(key_mask_1).eq(0))
            assert torch.all(accum_grad.masked_select(~key_mask_1).ne(0))

    assert torch.all(query.grad.ne(0))

def test_masking_output(key, query, key_mask, init_accumulator):
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    batch_attention, batch_state = kernel(key, query, key_mask=key_mask,
                                          state=init_accumulator)

    if key_mask is None:
        assert torch.all(batch_attention.ne(0))
    else:
        key_mask_1 = key_mask.unsqueeze(1) 
        assert torch.all(batch_attention.masked_select(key_mask_1).eq(0))
        assert torch.all(batch_attention.masked_select(~key_mask_1).ne(0))

    assert torch.allclose(batch_attention.sum(2), 
            torch.FloatTensor(batch_attention.size()[:2]).fill_(1.))

    assert torch.all(query.grad.ne(0))

def test_masking_output(key, query, key_mask, init_accumulator):
    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    batch_attention, batch_state = kernel(key, query, key_mask=key_mask,
                                          state=init_accumulator)

    if key_mask is None:
        assert torch.all(batch_attention.ne(0))
    else:
        key_mask_1 = key_mask.unsqueeze(1) 
        assert torch.all(batch_attention.masked_select(key_mask_1).eq(0))
        assert torch.all(batch_attention.masked_select(~key_mask_1).ne(0))

    assert torch.allclose(batch_attention.sum(2), 
            torch.FloatTensor(batch_attention.size()[:2]).fill_(1.))

def test_limiting_attention(key, query, key_length):

    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    query_1 = query[:1]

    state = None
    steps = 25
    for i in range(steps):
        attention, state = kernel(key, query_1, state=state)

    ref_sum = torch.FloatTensor(key.size(0), 1).fill_(steps)
    assert torch.allclose(state["accumulator"].sum(2), ref_sum)

    kernel = nnsum2.attention.AccumulatingBiLinearKernel()
    _, state = kernel(key, query_1)
    attention, state = kernel(key, query_1, 
        state={"accumulator": state["accumulator"] + 1000000000})
    
    uniform = torch.FloatTensor(
        key.size(0), 1, key_length).fill_(1 / key_length)
    assert torch.allclose(attention, uniform)
