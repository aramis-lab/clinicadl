import pytest
import torch


@pytest.mark.gpu
def test_gpu():
    assert torch.cuda.is_available()


@pytest.mark.multi_gpu
def test_multi_gpu():
    assert torch.cuda.device_count() > 1
