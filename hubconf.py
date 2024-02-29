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
