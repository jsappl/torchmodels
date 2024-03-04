"""Utilities to be reused across tests."""

import torch
import torch.nn as nn

EPS: float = 1e-6


def model_optimization_test(
        model: nn.Module, input_: torch.Tensor, target: torch.Tensor, tol: float = 1e-3, single_tol: float = 1e-3):
    """Test optimizing the model parameters.

    Args:
        model: An instantiated PyTorch model.
        input_: A input sample to be passed to the `model`.
        target: An expected output sample.
        tol: Mean norm of parameters should change at least this much.
        single_tol: Norm of each single parameter tensor should change at least this much.
    """
    old_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=10)
    optimizer.zero_grad()

    output = model(input_)
    assert output.size() == output.size()

    loss = nn.MSELoss()(output, target)
    assert loss.item() > 0
    loss.backward()

    assert all([parameter.grad is not None for parameter in model.parameters()])
    optimizer.step()
    changes = [
        nn.MSELoss()(parameter_old, parameter_new)
        for (parameter_old, parameter_new) in zip(old_parameters, model.parameters(), strict=True)
    ]

    assert sum(changes) / len(changes) > tol
    zero_grads = [torch.norm(parameter.grad).item() <= EPS for parameter in model.parameters()]
    for (norm_parameter, grad_is_zero) in zip(changes, zero_grads, strict=True):
        if grad_is_zero:
            assert norm_parameter > single_tol
