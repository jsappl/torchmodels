"""Entrypoints for generic PyTorch models."""

from typing import TYPE_CHECKING

import torchmodels

if TYPE_CHECKING:
    import torch.nn as nn

dependencies = ["torch"]


def multilayer_perceptron(neurons: list[int]) -> "nn.Module":
    """Load the multilayer perceptron model architecture.

    Args:
        neurons: Specify the number of neurons for each layer.

    Returns:
        An instance of the `torchmodels.MultilayerPerceptron` class.
    """
    return torchmodels.MultilayerPerceptron(neurons)


def unet(in_channels: int, out_channels: int, features: int) -> "nn.Module":
    """Load the U-Net model architecture.

    Args:
        in_channels: Color channels of the input image.
        out_channels: Color channels of the output image.
        features: Number of feature channels double for each layer.

    Returns:
        An instance of the `torchmodels.UNet` class.
    """
    return torchmodels.UNet(in_channels, out_channels, features)
