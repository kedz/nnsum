import pytest
import torch
import torch.nn as nn
from nnsum.seq2seq.cross_entropy_loss import CrossEntropyLoss
from nnsum.seq2seq import SearchState


@pytest.fixture(scope="function")
def logits():
    return nn.Parameter(torch.FloatTensor(3, 4, 5).normal_())

@pytest.fixture(scope="module")
def targets():
    return torch.LongTensor(4,3).random_(0, 5)

@pytest.fixture(scope="module")
def target_mask(request):
    if request.param:
        return torch.ByteTensor([[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0]])

@pytest.mark.parametrize("target_mask", [True, False], indirect=True)
def test_backprop(logits, targets, target_mask):

    loss_func = CrossEntropyLoss()
    forward_state = SearchState(target_logits=logits)
    batch = {"target_output_features": {"tokens": targets},
             "target_mask": target_mask}
    loss = loss_func(forward_state, batch)
    loss.backward()

    for step in range(logits.size(0)):
        for batch in range(logits.size(1)):
            if target_mask is None or target_mask[batch, step].item() == 0:
                assert torch.all(logits.grad[step, batch].ne(0))
            else:
                assert torch.all(logits.grad[step, batch].eq(0))

def test_accumulation():
    loss_func = CrossEntropyLoss()
    batch_size = 2
    steps = 3
    vsize = 5
    exc_str = "Must have processed at least one batch."

    for epoch in range(2):
        losses = []
        for i in range(4):
            logits = torch.FloatTensor(steps, batch_size, vsize).normal_()
            forward_state = SearchState(target_logits=logits)
            targets = torch.LongTensor(batch_size, steps).random_(0, vsize)
            batch = {"target_output_features": {"tokens": targets}}
            loss = loss_func(forward_state, batch)
            losses.append(loss.item() * batch_size * steps)

        expected_mean = torch.tensor(sum(losses) / (batch_size * steps * 4))
        accumulator_mean = torch.tensor(loss_func.mean())

        assert torch.allclose(expected_mean, accumulator_mean)
        loss_func.reset()
        with pytest.raises(RuntimeError) as excinfo:
            loss_func.mean()
            assert exc_str == str(excinfo.value)

def test_uniform_logits_is_max_entropy():
    logits = nn.Parameter(torch.FloatTensor(1, 1, 5).fill_(0))
    targets = torch.LongTensor([[0]])
    batch = {"target_output_features": {"tokens": targets}}
    loss_func = CrossEntropyLoss()
    forward_state = SearchState(target_logits=logits)
    loss = loss_func(forward_state, batch)
    assert torch.allclose(loss, torch.log(torch.tensor(5.)))

@pytest.mark.parametrize("target_mask", [True, False], indirect=True)
def test_singleton_matches_batch(logits, targets, target_mask):

    loss_func = CrossEntropyLoss()
    forward_state = SearchState(target_logits=logits)
    batch = {"target_output_features": {"tokens": targets},
             "target_mask": target_mask}
    batch_loss = loss_func(forward_state, batch)
 
    losses = []
    num_els = 0
    for step in range(targets.size(1)):
        for batch in range(targets.size(0)):
            logits_el = logits[step:step+1,batch:batch+1]
            targets_el = targets[batch:batch+1,step:step+1]
            if target_mask is not None:
                target_mask_el = target_mask[batch:batch+1,step:step+1]
                if target_mask_el.item() == 1:
                    continue
            else:
                target_mask_el = None
            batch_el = {"target_output_features": {"tokens": targets_el},
                        "target_mask": target_mask_el}
            search_state_el = SearchState(target_logits=logits_el)
            loss_el = loss_func(search_state_el, batch_el)
            losses.append(loss_el)
            num_els += 1
    
    el_loss = sum(losses) / num_els

    assert torch.allclose(el_loss, batch_loss)
