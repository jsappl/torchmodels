"""Configure/customize pytest."""
import pytest
import torch


def pytest_generate_tests(metafunc):
    """Parametrize tests."""
    if "device" in metafunc.fixturenames:
        if torch.cuda.is_available():
            metafunc.parametrize("device", ["cpu", "cuda"], indirect=True)
        else:
            metafunc.parametrize("device", ["cpu"], indirect=True)


@pytest.fixture
def device(request):
    """Pytorch devices."""
    return torch.device(request.param)
