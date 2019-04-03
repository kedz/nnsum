import pytest
import torch
import torch.nn as nn
from nnsum2.layers import ConvMaxPoolingSeq1D
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
        "batch_first": batch_first,
        "activation": "ReLU",
    }
    module = ConvMaxPoolingSeq1D(**hyperparameters)
    module.kernel_bias.data.fill_(1)
    module.eval()

    if batch_first:
        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()

    outputs, outputs_mask = module(inputs)

    assert outputs.dim() == 2
    assert outputs.size(0) == batch_size
    assert outputs.size(1) == output_features
    assert outputs_mask is None

@pytest.mark.parametrize("kernel_width", [1, 2, 3])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("activation", [None, "ReLU", "Tanh", "Sigmoid"])
def test_singleton_matches_batch(kernel_width, batch_first, padding, 
                                 activation):
    batch_size = 3
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
        "batch_first": batch_first,
        "activation": activation,
    }
    module = ConvMaxPoolingSeq1D(**hyperparameters)
    module.kernel_bias.data.fill_(-.05)
    module.eval()

    lengths = [3, 5, 4]

    if batch_first:
        inputs_mask = torch.ByteTensor(batch_size, seq_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[i,length:].fill_(1)
        inputs = torch.FloatTensor(
            batch_size, seq_size, input_features).normal_()
    else:
        inputs_mask = torch.ByteTensor(seq_size, batch_size).fill_(0)
        for i, length in enumerate(lengths):
            inputs_mask[length:,i].fill_(1)
        inputs = torch.FloatTensor(
            seq_size, batch_size, input_features).normal_()

    batch_inputs = nn.Parameter(inputs)   

    batch_outputs, _ = module(batch_inputs, inputs_mask)
    batch_outputs.sum().backward()

    batch_grads = [("inputs", batch_inputs.grad.clone()),]
    batch_inputs.grad.data.fill_(0)
    for name, param in module.named_parameters():
        batch_grads.append((name, param.grad.clone()))
        param.grad.data.fill_(0)

    singleton_outputs = []
    for i, length in enumerate(lengths):
        if batch_first:
            inputs_i = batch_inputs[i:i+1,:length]
        else:    
            inputs_i = batch_inputs[:length,i:i+1]
        outputs_i, _ = module(inputs_i)
        singleton_outputs.append(outputs_i)
    singleton_outputs = torch.cat(singleton_outputs, 0)
    singleton_outputs.sum().backward()
    singleton_grads = [("inputs", batch_inputs.grad.clone()),]
    batch_inputs.grad.data.fill_(0)
    for name, param in module.named_parameters():
        singleton_grads.append((name, param.grad.clone()))
        param.grad.data.fill_(0)
    
    assert torch.allclose(singleton_outputs, batch_outputs)

    for (name1, grad1), (name2, grad2) in zip(batch_grads, singleton_grads):
        assert name1 == name2
        assert torch.allclose(grad1, grad2, atol=1e-6, rtol=0)

@pytest.mark.parametrize("dropout", [0, .1, .5, .75])
def test_dropout(dropout):

    batch_size = 1
    seq_size = 5
    input_features = 16
    output_features = 10000
    hyperparameters = {
        "input_features": input_features,
        "output_features": output_features,
        "kernel_width": 3,
        "activation": None,
        "dropout": dropout,
    }
    module = ConvMaxPoolingSeq1D(**hyperparameters)
    module.initialize_parameters()

    inputs = torch.FloatTensor(
        batch_size, seq_size, input_features).normal_()

    module.train()
    train_outputs, _ = module(inputs)

    num_zero = train_outputs.eq(0.).sum().item()
    num_el = train_outputs.numel()
    actual_dropout_rate = num_zero / num_el
    assert math.fabs(actual_dropout_rate - dropout) < 1e-2

    module.eval()
    eval_outputs, _ = module(inputs)
    assert torch.all(eval_outputs.ne(0.))
