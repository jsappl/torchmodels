"""Testing utilities."""
import torch
from torch.nn import MSELoss

EPS = 1e-6


def modeloptimizationtest(model, inputsample, outputsample, tol: float = 1e-1):
    """Test model optimization.

    Args:
        model: the model should be `torch.nn.Module` or a superclass of it
        inputsample: one sample of the model inputs output will be generated using `model(inputsample)`
        outputsample: one sample for the outputs
        tol: norm of parameters should change at least this much
    """
    old_parameters = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=10)
    opt.zero_grad()
    output = model(inputsample)
    # MSELoss would still work if they have different shapes
    assert output.size() == outputsample.size()
    loss = MSELoss()(output, outputsample)
    # if the loss is already 0 we can not optimize
    assert loss.item() > 0
    loss.backward()
    # all parameters should have gradients
    assert all([p.grad is not None for p in model.parameters()])
    opt.step()
    changes = [MSELoss()(pold, pnew) for (pold, pnew) in zip(old_parameters, model.parameters(), strict=True)]
    # some should change
    assert sum(changes) / len(changes) > tol
    # all with gradients should change
    zero_grads = [torch.norm(p.grad).item() <= EPS for p in model.parameters()]
    for (paramnorm, gradiszero) in zip(changes, zero_grads, strict=True):
        if gradiszero:
            assert paramnorm > tol
