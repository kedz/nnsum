import pytest
import torch
import torch.nn as nn
from nnsum2.layers import MaxPoolingSeq1D
import math


def pname(name):
    def namer(value):
        return "{}={}".format(name, str(value))
    return namer

def compute_output_size(seq_size, kernel_width, stride, padding, dilation): 
    if kernel_width is None:
        kernel_width = seq_size
    return math.floor(
        (
            seq_size + (2 * padding) - (dilation * (kernel_width - 1)) - 1
        ) / stride
        + 1
    )

@pytest.mark.parametrize("batch_size", [3, 5], ids=pname("batch_size"))
@pytest.mark.parametrize("kernel_width", [None, 2, 3], 
                         ids=pname("kernel_width"))
@pytest.mark.parametrize("batch_first", [True, False], 
                         ids=pname("batch_first"))
@pytest.mark.parametrize("squeeze_singleton", [True, False],
                         ids=pname("squeeze_singleton"))
def test_forward(batch_size, kernel_width, batch_first, squeeze_singleton):
    seq_size = 5
    input_features = output_features = 4
    dilation = 1
    padding = 0
    stride = kernel_width if kernel_width is not None else seq_size
    hyperparameters = {
        "kernel_width": kernel_width,
        "batch_first": batch_first,
        "squeeze_singleton": squeeze_singleton,
    }
    module = MaxPoolingSeq1D(**hyperparameters)

    if batch_first:
        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()

    outputs, outputs_mask = module(inputs)

    expected_sequence_size = compute_output_size(
        seq_size, kernel_width, stride, padding, dilation)

    if squeeze_singleton and expected_sequence_size == 1:
        assert outputs.dim() == 2
        assert outputs.size(0) == batch_size
        assert outputs.size(1) == output_features
    else:
        assert outputs.dim() == 3
        assert outputs.size(0) == batch_size
        assert outputs.size(1) == expected_sequence_size
        assert outputs.size(2) == output_features
    assert outputs_mask is None


@pytest.mark.parametrize("kernel_width", [None, 2, 3], 
                         ids=pname("kernel_width"))
@pytest.mark.parametrize("batch_first", [True, False], 
                         ids=pname("batch_first"))
@pytest.mark.parametrize("squeeze_singleton", [True, False],
                         ids=pname("squeeze_singleton"))
def test_safe_masking(kernel_width, batch_first, squeeze_singleton):

    batch_size = 3
    seq_size = 5
    input_features = output_features = 4
    dilation = 1
    stride = kernel_width if kernel_width is not None else seq_size
    padding = 0

    hyperparameters = {
        "kernel_width": kernel_width,
        "batch_first": batch_first,
        "squeeze_singleton": squeeze_singleton,
    }
    module = MaxPoolingSeq1D(**hyperparameters)
    lengths = [3, 5, 4]

    expected_sequence_size = compute_output_size(
        seq_size, kernel_width, stride, padding, dilation)

    if batch_first:
        inputs_mask = torch.ByteTensor(batch_size, seq_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[i,length:].fill_(1)

        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()
        inputs_mask = torch.ByteTensor(seq_size, batch_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[length:,i].fill_(1)

    inputs = nn.Parameter(inputs)        
    safe_inputs = inputs.masked_fill(inputs_mask.unsqueeze(2), float("-inf")) 
    outputs, outputs_mask = module(inputs, inputs_mask)
    safe_outputs, safe_outputs_mask = module(safe_inputs, inputs_mask)

    expected_mask = []
    for i, length in enumerate(lengths):
        out_len_i = compute_output_size(
            length, kernel_width, stride, padding, dilation) 
        m = torch.ByteTensor(1, expected_sequence_size).fill_(0)
        m[0, out_len_i:].fill_(1)
        expected_mask.append(m)
    expected_mask = torch.cat(expected_mask, 0)
    if squeeze_singleton and expected_mask.size(1) == 1:
        expected_mask = expected_mask.view(-1)

    assert torch.all(expected_mask == outputs_mask)
    assert torch.all(expected_mask == safe_outputs_mask)
    assert torch.allclose(safe_outputs, outputs)

    singleton_outputs = []
    module._squeeze_singleton = False
    for i, length in enumerate(lengths):
        if batch_first:
            inputs_i = inputs[i:i+1,:length]
        else:
            inputs_i = inputs[:length,i:i+1]
        outputs_i, _ = module(inputs_i)

        diff = expected_sequence_size - outputs_i.size(1)
        if outputs_i.dim() == 3 and diff > 0:
            outputs_i = torch.cat(
                [outputs_i,
                 outputs_i.new(1, diff, output_features).fill_(0)],
                1)
        singleton_outputs.append(outputs_i)
    
    singleton_outputs = torch.cat(singleton_outputs, 0)
    if squeeze_singleton and singleton_outputs.size(1) == 1:
        singleton_outputs = singleton_outputs.squeeze(1)

    assert torch.allclose(singleton_outputs, safe_outputs)
    assert torch.allclose(singleton_outputs, outputs)
    module._squeeze_singleton = squeeze_singleton
    
    singleton_grads = []
    singleton_outputs.masked_fill(
        outputs_mask.unsqueeze(-1), 0).sum().backward()
    for name, params in module.named_parameters():
        singleton_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    outputs, _ = module(inputs, inputs_mask)
    outputs.masked_fill(outputs_mask.unsqueeze(-1), 0).sum().backward()
    outputs_grads = []
    for name, params in module.named_parameters():
        outputs_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    for (name1, grad1), (name2, grad2) in zip(singleton_grads, outputs_grads):
        assert name1 == name2
        assert torch.allclose(grad1, grad2)

    outputs_safe, _ = module(safe_inputs, inputs_mask)
    outputs_safe.masked_fill(outputs_mask.unsqueeze(-1), 0).sum().backward()
    outputs_safe_grads = []
    for name, params in module.named_parameters():
        outputs_safe_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    for (name1, grad1), (name2, grad2) in zip(singleton_grads, 
                                              outputs_safe_grads):
        assert name1 == name2
        assert torch.allclose(grad1, grad2)
