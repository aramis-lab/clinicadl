import os
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from clinicadl.train import ComputationalConfig


def test_ComputationalConfig():
    config = ComputationalConfig(
        gpu=False,
        non_blocking=False,
        amp=False,
        channels_last=False,
        seed=0,
        deterministic=True,
    )

    assert not config.gpu
    assert not config.non_blocking
    assert not config.amp
    assert not config.channels_last
    assert config.seed == 0
    assert config.device == torch.device("cpu")
    scaler = config.get_scaler()
    assert not scaler._enabled
    assert config.deterministic

    config.amp = True
    scaler = config.get_scaler()
    assert scaler._enabled
    assert scaler._device == "cpu"

    with pytest.raises(ValidationError):
        config.seed = -1

    config = ComputationalConfig(gpu=True)
    with pytest.raises(AssertionError, match="No GPU with CUDA available."):
        config.check_device()


@patch.dict(os.environ, {}, clear=True)
def test_global_variables():
    config = ComputationalConfig()
    assert config.seed is None
    assert not config.deterministic
    os.environ["CLINICADL_GLOBAL_SEED"] = "1"
    os.environ["CLINICADL_DETERMINISTIC"] = "true"
    config = ComputationalConfig()
    assert config.deterministic
    assert config.seed == 1
    config = ComputationalConfig(seed=2, deterministic=False)
    assert not config.deterministic
    assert config.seed == 2


@pytest.mark.gpu
def test_gpu():
    config = ComputationalConfig(gpu=True, amp=True)
    config.check_device()
    assert config.gpu
    assert config.device == torch.device("cuda")
    scaler = config.get_scaler()
    assert scaler._enabled
    assert scaler._device == "cuda"
