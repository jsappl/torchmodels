"""Test the multilayer perceptron implementation."""

import pytest
import torch
import torch.nn as nn

from torchmodels import MultilayerPerceptron

from .utils import model_optimization_test


@pytest.fixture
def model():
    """Create an instance of the MultilayerPerceptron for testing."""
    neurons = [2, 2, 1]
    return MultilayerPerceptron(neurons)


def test_forward_pass(model, device):
    """Test the forward pass of the MultilayerPerceptron."""
    model = model.to(device)
    input_tensor = torch.randn((4, 2), device=device)
    output = model.forward(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 1)


def test_invalid_neurons():
    """Test if invalid input shape raises an error."""
    with pytest.raises(AssertionError):
        MultilayerPerceptron([1])


def test_layers(model):
    """Test if the layers are constructed properly."""
    expected_layers = [nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)]

    for index, layer in enumerate(model.layers):
        assert isinstance(layer, type(expected_layers[index]))

        if isinstance(layer, nn.Linear):
            assert layer.in_features == expected_layers[index].in_features
            assert layer.out_features == expected_layers[index].out_features


def test_optimization(model, device):
    """Test optimization."""
    input_ = torch.randn(4, 2, device=device)
    output = torch.rand(4, 1, device=device)
    model = model.to(device)
    model_optimization_test(model, input_, output, 1e-3, 1e-6)
