import pytest
import torch
from nnsum2.layers import ConvSeq1D
import math


@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize("kernel_width", [1, 2, 3])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_forward(batch_size, kernel_width, batch_first, padding):
    seq_size = 5
    input_features = 16
    output_features = 8
    dilation = 1
    stride = 1
    hyperparameters = {
        "input_features": input_features,
        "output_features": output_features,
        "kernel_width": kernel_width,
        "padding": padding,
        "batch_first": batch_first
    }
    module = ConvSeq1D(**hyperparameters)

    expected_output_dim0_size = batch_size
    expected_output_dim1_size = math.floor(
        (seq_size + padding * 2 - dilation * (kernel_width - 1) - 1) / stride
        + 1
    )
    expected_output_dim2_size = output_features
   
    if batch_first:
        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()

    outputs, outputs_mask = module(inputs)

    assert outputs.size(0) == expected_output_dim0_size
    assert outputs.size(1) == expected_output_dim1_size
    assert outputs.size(2) == expected_output_dim2_size
    assert outputs_mask is None

@pytest.mark.parametrize("kernel_width", [1, 2, 3])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_masking_safe(kernel_width, batch_first, padding):
    
    batch_size = 3
    seq_size = 5
    output_features = 8

    seq_size = 5
    input_features = 4
    output_features = 8
    dilation = 1
    stride = 1
    hyperparameters = {
        "input_features": input_features,
        "output_features": output_features,
        "kernel_width": kernel_width,
        "padding": padding,
        "batch_first": batch_first
    }
    module = ConvSeq1D(**hyperparameters)
    module.kernel_bias.data.fill_(1)

    out_seq_size = math.floor(
        (seq_size + padding * 2 - dilation * (kernel_width - 1) - 1) / stride
        + 1
    )
    lengths = [3, 5, 4]

    if batch_first:
        mask = torch.ByteTensor(batch_size, seq_size).fill_(0)
        for i, length in enumerate(lengths):
            mask[i,length:].fill_(1)

        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()
        mask = torch.ByteTensor(seq_size, batch_size).fill_(0)
        for i, length in enumerate(lengths):
            mask[length:,i].fill_(1)
        
    safe_inputs = inputs.masked_fill(mask.unsqueeze(2), 0)
     
    outputs, outputs_mask = module(inputs, mask)
    safe_outputs, safe_mask = module(safe_inputs, mask)

    expected_mask = []
    for i, length in enumerate(lengths):
        out_length = math.floor(
            (length + padding * 2 - dilation * (kernel_width - 1) - 1) / stride
            + 1
        )
        m = torch.ByteTensor(1, out_seq_size).fill_(0)
        m[0,out_length:].fill_(1)
        expected_mask.append(m)
    expected_mask = torch.cat(expected_mask, 0)

    assert torch.all(expected_mask == outputs_mask)
    assert torch.all(expected_mask == safe_mask)
    assert torch.allclose(safe_outputs, outputs)

    singleton_outputs = []
    for i, length in enumerate(lengths):
        if batch_first:
            inputs_i = inputs[i:i+1,:length]
        else:
            inputs_i = inputs[:length,i:i+1]
        outputs_i, _ = module(inputs_i)
        diff = out_seq_size - outputs_i.size(1)
        if diff > 0:
            outputs_i = torch.cat(
                [outputs_i,
                 outputs_i.new(1, diff, output_features).fill_(0)],
                1)
        singleton_outputs.append(outputs_i)
    
    singleton_outputs = torch.cat(singleton_outputs, 0)

    assert torch.allclose(singleton_outputs, safe_outputs)

    singleton_grads = []
    singleton_outputs.masked_fill(
        outputs_mask.unsqueeze(2), 0).sum().backward()
    for name, params in module.named_parameters():
        singleton_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    outputs, _ = module(inputs, mask)
    outputs.masked_fill(outputs_mask.unsqueeze(2), 0).sum().backward()
    outputs_grads = []
    for name, params in module.named_parameters():
        outputs_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    for (name1, grad1), (name2, grad2) in zip(singleton_grads, outputs_grads):
        assert name1 == name2
        assert torch.allclose(grad1, grad2)

    outputs_safe, _ = module(safe_inputs, mask)
    outputs_safe.masked_fill(outputs_mask.unsqueeze(2), 0).sum().backward()
    outputs_safe_grads = []
    for name, params in module.named_parameters():
        outputs_safe_grads.append((name, params.grad.clone()))
        params.grad.fill_(0)

    for (name1, grad1), (name2, grad2) in zip(singleton_grads, 
                                              outputs_safe_grads):
        assert name1 == name2
        assert torch.allclose(grad1, grad2)
