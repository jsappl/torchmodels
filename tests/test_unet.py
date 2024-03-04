"""Test the U-Net implementation."""

import pytest
import torch
import torch.nn as nn

from torchmodels import UNet

from .utils import model_optimization_test

SIZE: int = 32


@pytest.fixture
def model():
    """Create an instance of the UNet model for testing."""
    return UNet(in_channels=1, out_channels=1, features=2)


def test_forward_pass(model, device):
    """Test the forward pass of the UNet model."""
    model = model.to(device)
    input_ = torch.randn(1, 1, SIZE, SIZE, device=device)
    output = model.forward(input_)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, SIZE, SIZE)


def test_initialization():
    """Test if the UNet model is initialized correctly."""
    model = UNet(in_channels=1, out_channels=1, features=2)

    assert isinstance(model, nn.Module)


def test_layers(model):
    """Test the layers of the model architecture."""
    assert hasattr(model, "bottleneck")

    for index in range(1, 5):
        assert hasattr(model, f"pool{index}")
        assert hasattr(model, f"deconv{index}")

    for index in range(1, 9):
        assert hasattr(model, f"doubleconv{index}")


def test_double_convolution(model, device):
    """Test the double convolution layer."""
    in_channels = 3
    out_channels = 1
    doubleconv = model._double_convolution(in_channels, out_channels).to(device)

    input_ = torch.randn(1, in_channels, 8, 8, device=device)
    output = doubleconv(input_)

    assert output.shape == (1, out_channels, 8, 8)
    assert torch.min(output).item() >= 0


def test_optimization(model, device):
    """Test optimization."""
    input_ = torch.randn(1, 1, SIZE, SIZE, device=device)
    output = torch.rand(1, 1, SIZE, SIZE, device=device)
    model = model.to(device)
    model_optimization_test(model, input_, output)
