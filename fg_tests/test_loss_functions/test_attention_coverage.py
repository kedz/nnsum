import pytest
import torch
import torch.nn as nn
from nnsum.seq2seq.attention_coverage import AttentionCoverage
from nnsum.attention import DotAttention
from nnsum.seq2seq import SearchState


@pytest.fixture(scope="function")
def context():
    return nn.Parameter(torch.FloatTensor(3, 4, 5).normal_())

@pytest.fixture(scope="module")
def source_mask(request):
    if request.param:
        return torch.ByteTensor([[0,0,0,0],[0,0,0,1],[1,1,1,1]])
    else:
        return None

@pytest.fixture(scope="function")
def target():
    return nn.Parameter(torch.FloatTensor(2, 3, 5).normal_())

@pytest.fixture(scope="module")
def target_mask(request):
    if request.param:
        return torch.ByteTensor([[0,0,0],[1,0,1]]).t()
    else:
        return None

@pytest.mark.parametrize("source_mask", [True, False], indirect=True)
@pytest.mark.parametrize("target_mask", [True, False], indirect=True)
@pytest.mark.parametrize("iterative", [True, False])
def test_backprop(context, source_mask, target, target_mask, iterative):

    amod = DotAttention(5, 5, temp=1.0, compute_composition=False)
    attention = amod(context, target, context_mask=source_mask)

    forward_state = SearchState(context_attention=attention)
    
    batch = {"source_mask": source_mask, "target_mask": target_mask}
    loss_func = AttentionCoverage(iterative=iterative)
    loss = loss_func(forward_state, batch)

    loss.backward()
    
    for step in range(context.size(1)):
        for batch in range(context.size(0)):
            if source_mask is None or source_mask[batch,step].item() == 0:
                assert torch.all(context.grad[batch, step].ne(0.))
            else:
                assert torch.all(context.grad[batch, step].eq(0.))

    if source_mask is not None:
        null_mask = torch.all(source_mask, dim=1)
    else:
        null_mask = None

    for step in range(target.size(0)):
        for batch in range(target.size(1)):
            if (target_mask is None or target_mask[batch, step].item() == 0) \
                    and (null_mask is None or null_mask[batch] == 0):
                assert torch.all(target.grad[step, batch].ne(0))
            else:
                assert torch.all(target.grad[step, batch].eq(0))

def test_accumulation():
    
    loss_func = AttentionCoverage()
    batch_size = 2
    exc_str = "Must have processed at least one batch."

    for epoch in range(2):
        losses = []
        for i in range(4):
            attention = torch.softmax(
                torch.FloatTensor(3, batch_size, 4).normal_(), dim=2)
            forward_state = SearchState(context_attention=attention)
            loss = loss_func(forward_state, {})
            losses.append(loss.item())

        expected_mean = torch.tensor(sum(losses) / (len(losses) * batch_size))
        accumulator_mean = torch.tensor(loss_func.mean())

        assert torch.allclose(expected_mean, accumulator_mean)
        loss_func.reset()
        with pytest.raises(RuntimeError) as excinfo:
            loss_func.mean()
            assert exc_str == str(excinfo.value)
