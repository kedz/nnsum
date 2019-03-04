import pytest
import torch
import torch.nn as nn
from nnsum.seq2seq.pointer_generator_cross_entropy_loss import (
    PointerGeneratorCrossEntropyLoss)
from nnsum.seq2seq import SearchState


@pytest.fixture(scope="module")
def batch_size():
    return 3

@pytest.fixture(scope="module")
def step_size():
    return 4

@pytest.fixture(scope="module")
def vocab_size():
    return 6 + 4

@pytest.fixture(scope="module")
def extended_vocab_size(vocab_size):
    return vocab_size + 5

@pytest.fixture(scope="function")
def pointer():
    return nn.Parameter(torch.FloatTensor(3, 4, 5).normal_())

@pytest.fixture(scope="module")
def extended_targets(batch_size, step_size, extended_vocab_size):
    targets = torch.LongTensor(batch_size, step_size)
    return targets.random_(4, extended_vocab_size)

@pytest.fixture(scope="module")
def generator_targets(vocab_size, extended_vocab_size, extended_targets):
    mask = extended_targets.ge(vocab_size)
    generator_targets = extended_targets.masked_fill(mask, 1)
    return generator_targets

@pytest.fixture(scope="module")
def switch_prob(batch_size, step_size):
    return torch.sigmoid(torch.FloatTensor(step_size, batch_size).normal_())

@pytest.fixture(scope="module")
def pointer_probs(switch_prob, batch_size, step_size, extended_vocab_size):
    logits = torch.FloatTensor(step_size, batch_size, extended_vocab_size)
    logits.normal_()
    unweighted_probs = torch.softmax(logits, dim=2)
    return nn.Parameter(switch_prob.unsqueeze(2) * unweighted_probs)

@pytest.fixture(scope="module")
def generator_probs(switch_prob, batch_size, step_size, vocab_size):
    logits = torch.FloatTensor(step_size, batch_size, vocab_size).normal_()
    unweighted_probs = torch.softmax(logits, dim=2)
    return nn.Parameter((1 - switch_prob.unsqueeze(2)) * unweighted_probs)

@pytest.fixture(scope="function")
def target_mask(request, batch_size, step_size):

    if request.param:
        m = torch.ByteTensor(batch_size, step_size).zero_()
        if step_size == 1:
            raise Exception("Step size must be at least 1.")
        prefix_length = torch.LongTensor(batch_size).random_(0, step_size - 1)
        for i, l in enumerate(prefix_length):
            if l > 0:
                m[i,-l:].fill_(1)
        return m
    else:
        return None

@pytest.mark.parametrize("target_mask", [True, False], indirect=True)
def test_backprop(extended_targets, generator_targets, target_mask,
                  pointer_probs, generator_probs, batch_size, step_size):

    loss_func = PointerGeneratorCrossEntropyLoss()
    forward_state = SearchState(pointer_probability=pointer_probs,
                                generator_probability=generator_probs)
    batch = {"target_output_features": {"tokens": generator_targets},
             "copy_targets": extended_targets, "target_mask": target_mask}

    loss = loss_func(forward_state, batch)
    loss.backward()
    
    for s in range(step_size):
        for b in range(batch_size):
            gtgt = generator_targets[b, s].item()
            ptgt = extended_targets[b, s].item()

            if target_mask is None or target_mask[b, s].item() == 0:
                # We are only coping from extended vocab and not generating
                # Gradient should be 0 for generator dist. 
                if gtgt != ptgt:
                    assert torch.all(generator_probs.grad[s, b].eq(0))
                    assert torch.all(pointer_probs.grad[s, b, ptgt].ne(0))
                    assert torch.all(pointer_probs.grad[s, b, :ptgt].eq(0))
                    assert torch.all(pointer_probs.grad[s, b, ptgt+1:].eq(0))

                # We are both copying and generating so there should be 
                # gradients flowing back to both distributions.
                else:
                    assert torch.all(generator_probs.grad[s, b, gtgt].ne(0))
                    assert torch.all(generator_probs.grad[s, b, :gtgt].eq(0))
                    assert torch.all(generator_probs.grad[s, b, gtgt+1:].eq(0))
                    assert torch.all(pointer_probs.grad[s, b, ptgt].ne(0))
                    assert torch.all(pointer_probs.grad[s, b, :ptgt].eq(0))
                    assert torch.all(pointer_probs.grad[s, b, ptgt+1:].eq(0))
            else:
                assert torch.all(generator_probs.grad[s, b].eq(0))
                assert torch.all(pointer_probs.grad[s, b].eq(0))

def test_accumulation():
    loss_func = PointerGeneratorCrossEntropyLoss()
    batch_size = 2
    steps = 3
    vsize = 10
    evsize = vsize + 5
    exc_str = "Must have processed at least one batch."

    for epoch in range(2):
        losses = []
        for i in range(4):
            ptr_targets = torch.LongTensor(batch_size, steps).random_(
                0, evsize)
            gen_targets = ptr_targets.masked_fill(ptr_targets.ge(vsize), 1) 

            ptr_probs = torch.softmax(
                torch.FloatTensor(steps, batch_size, evsize).normal_(), dim=2)
            gen_probs = torch.softmax(
                torch.FloatTensor(steps, batch_size, vsize).normal_(), dim=2)

            forward_state = SearchState(pointer_probability=ptr_probs,
                                        generator_probability=gen_probs)
            batch = {"target_output_features": {"tokens": gen_targets},
                     "copy_targets": ptr_targets}
            loss = loss_func(forward_state, batch)
            losses.append(loss.item() * batch_size * steps)

        expected_mean = torch.tensor(sum(losses) / (batch_size * steps * 4))
        accumulator_mean = torch.tensor(loss_func.mean())

        assert torch.allclose(expected_mean, accumulator_mean)
        loss_func.reset()
        with pytest.raises(RuntimeError) as excinfo:
            loss_func.mean()
            assert exc_str == str(excinfo.value)

@pytest.mark.parametrize("target_mask", [True, False], indirect=True)
def test_singleton_matches_batch(extended_targets, generator_targets, 
                                 target_mask, pointer_probs, generator_probs, 
                                 batch_size, step_size):

    loss_func = PointerGeneratorCrossEntropyLoss()
    batch_forward_state = SearchState(pointer_probability=pointer_probs,
                                      generator_probability=generator_probs)
    batch = {"target_output_features": {"tokens": generator_targets},
             "copy_targets": extended_targets, "target_mask": target_mask}

    batch_loss = loss_func(batch_forward_state, batch)
 
    losses = []
    num_els = 0
    for s in range(step_size):
        for b in range(batch_size):
            if target_mask is not None:
                target_mask_el = target_mask[b:b+1,s:s+1]
                if target_mask_el.item() == 1:
                    continue
            else:
                target_mask_el = None
            ext_tgt_el = extended_targets[b:b+1,s:s+1]
            gen_tgt_el = generator_targets[b:b+1,s:s+1]
            batch_el = {"target_output_features": {"tokens": gen_tgt_el},
                        "copy_targets": ext_tgt_el,
                        "target_mask": target_mask_el}

            ptr_prob_el = pointer_probs[s:s+1,b:b+1]
            gen_prob_el = generator_probs[s:s+1,b:b+1]
            forward_state_el = SearchState(pointer_probability=ptr_prob_el,
                                           generator_probability=gen_prob_el)
            loss_el = loss_func(forward_state_el, batch_el)
            losses.append(loss_el)
            num_els += 1
             
    el_loss = sum(losses) / num_els
    assert torch.allclose(el_loss, batch_loss)
