import pytest
import torch
import nnsum2


def test_is_stateless():
    kernel = nnsum2.attention.BiLinearKernel()
    assert kernel.is_stateless == True

def test_sequential_matches_batch(key, query, key_mask, batch_size, 
                                  key_length, query_length):
    module = nnsum2.attention.BiLinearKernel()
    reward = torch.FloatTensor(batch_size, query_length, key_length).normal_()

    # Run batch mode.
    batch_attention, batch_state = module(key, query, key_mask=key_mask)
    assert batch_state is None
    (batch_attention * reward).mean().backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)

    # Run sequentially and collect back into batch.
    query_steps = query.split(1, dim=0)

    step_state = None
    step_attentions = []
    for query_step in query_steps:
        step_attention, step_state = module(key, query_step, key_mask=key_mask,
                                            state=step_state)
        assert step_state is None
        step_attentions.append(step_attention)

    step_attentions = torch.cat(step_attentions, dim=1)
    (step_attentions * reward).mean().backward()
    step_key_grad = key.grad.clone()
    step_query_grad = query.grad.clone()

    assert torch.allclose(step_key_grad, batch_key_grad)
    assert torch.allclose(step_query_grad, batch_query_grad)
    assert torch.allclose(step_attentions, batch_attention)

def test_singleton_matches_batch(key, query, key_mask, batch_size,
                                 key_length, query_length):
    module = nnsum2.attention.BiLinearKernel()
    reward = torch.FloatTensor(batch_size, query_length, key_length).normal_()

    # Run batch mode.
    batch_attention, batch_state = module(key, query, key_mask=key_mask)
    assert batch_state is None
    (batch_attention * reward).mean().backward()
    batch_key_grad = key.grad.clone()
    key.grad.fill_(0)
    batch_query_grad = query.grad.clone()
    query.grad.fill_(0)

    # Run singleton and collect back into batch.
    single_keys = key.split(1, dim=0)
    single_queries = query.split(1, dim=1)
    if key_mask is not None:
        single_key_masks = key_mask.split(1, dim=0)
    else:
        single_key_masks = [None] * len(single_keys)

    step_state = None
    single_attentions = []
    for single_key, single_query, single_key_mask in zip(
            single_keys, single_queries, single_key_masks):
        single_attention, step_state = module(
            single_key, single_query, key_mask=single_key_mask,
            state=step_state)
        assert step_state is None
        single_attentions.append(single_attention)

    single_attentions = torch.cat(single_attentions, dim=0)
    (single_attentions * reward).mean().backward()
    single_key_grad = key.grad.clone()
    single_query_grad = query.grad.clone()

    assert torch.allclose(single_key_grad, batch_key_grad)
    assert torch.allclose(single_query_grad, batch_query_grad)
    assert torch.allclose(single_attentions, batch_attention)

def test_key_masking_gradient(key, query, key_mask):
    module = nnsum2.attention.BiLinearKernel()
    batch_attention, batch_state = module(key, query, key_mask=key_mask)

    reward = torch.FloatTensor(batch_attention.size()).normal_()
    (batch_attention * reward).mean().backward()

    if key_mask is None:
        assert torch.all(key.grad.ne(0))

    else:
        key_mask_1 = key_mask.unsqueeze(2) 
        assert torch.all(key.grad.masked_select(key_mask_1).eq(0))
        assert torch.all(key.grad.masked_select(~key_mask_1).ne(0))

    assert torch.all(query.grad.ne(0))

def test_masking_output(key, query, key_mask):
    module = nnsum2.attention.BiLinearKernel()
    batch_attention, batch_state = module(key, query, key_mask=key_mask)
    assert batch_state is None

    if key_mask is None:
        assert torch.all(batch_attention.ne(0))
    else:
        key_mask_1 = key_mask.unsqueeze(1) 
        assert torch.all(batch_attention.masked_select(key_mask_1).eq(0))
        assert torch.all(batch_attention.masked_select(~key_mask_1).ne(0))

    assert torch.allclose(batch_attention.sum(2), 
            torch.FloatTensor(batch_attention.size()[:2]).fill_(1.))
