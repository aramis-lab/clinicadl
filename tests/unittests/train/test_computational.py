import pytest
import torch
from pydantic import ValidationError

from clinicadl.train import ComputationalConfig


def test_ComputationalConfig():
    config = ComputationalConfig(
        non_blocking=False, amp=False, channels_last=False, seed=0, checkpoint_every=10
    )

    assert not config.non_blocking
    assert not config.amp
    assert not config.channels_last
    assert config.seed == 0
    assert config.checkpoint_every == 10
    assert config.device == torch.device("cpu")
    scaler = config.get_scaler()
    assert not scaler._enabled

    config.amp = True
    scaler = config.get_scaler()
    assert scaler._enabled
    assert scaler._device == "cpu"

    with pytest.raises(ValidationError):
        config.gpu = True
    with pytest.raises(ValidationError):
        config.seed = -1
    with pytest.raises(ValidationError):
        config.checkpoint_every = 0


@pytest.mark.gpu
def test_gpu():
    config = ComputationalConfig(gpu=True, amp=True)
    assert config.gpu
    assert config.device == torch.device("gpu")
    scaler = config.get_scaler()
    assert scaler._enabled
    assert scaler._device == "gpu"
