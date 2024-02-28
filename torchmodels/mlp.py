"""A generic version of the multilayer perceptron (MLP) model class."""

import torch
import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    """Feedforward neural network consisting of fully connected neurons with ReLU activation."""

    def __init__(self, neurons: list[int]) -> None:
        """Initalize the layers in the model.

        Args:
            neurons: Specify the number of neurons for each layer.
        """
        assert len(neurons) > 1, "Provide at least input and output size"
        super().__init__()

        self.layers = nn.Sequential()

        for index, (neurons_in, neurons_out) in enumerate(zip(neurons[:-1], neurons[1:], strict=True)):
            self.layers.append(nn.Linear(neurons_in, neurons_out))

            if index != len(neurons) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """A single forward pass through the model.

        Args:
            input_: The input tensor to be processed.

        Returns:
            The multilayer perceptron model output.
        """
        return self.layers(input_)
