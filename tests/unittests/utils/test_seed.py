import os
import random
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from clinicadl.utils.seed import (
    pl_worker_init_function,
    seed_everything,
    seed_everything_context,
)


def test_seed_everything(caplog):
    assert "CLINICADL_GLOBAL_SEED" not in os.environ
    assert "CLINICADL_DETERMINISTIC" not in os.environ
    assert os.environ.get("PYTHONHASHSEED") != "10"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"

    with caplog.at_level("INFO"):
        seed_everything(seed=10)
    assert "Global seed set to 10" in caplog.text
    assert os.environ.get("PYTHONHASHSEED") == "10"
    assert os.environ.get("CLINICADL_GLOBAL_SEED") == "10"
    assert "CLINICADL_DETERMINISTIC" not in os.environ

    seed_everything(seed=11, deterministic=True)
    assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert torch.are_deterministic_algorithms_enabled()

    seed_everything(0)
    r1 = random.randint(0, 100)
    n1 = np.random.rand()
    t1 = torch.rand(1)

    seed_everything(0)
    r1_ = random.randint(0, 100)
    n1_ = np.random.rand()
    t1_ = torch.rand(1)

    seed_everything(1)
    r2 = random.randint(0, 100)
    n2 = np.random.rand()
    t2 = torch.rand(1)

    assert np.isclose(r1, r1_)
    assert np.isclose(n1, n1_)
    assert torch.isclose(t1, t1_)

    assert not np.isclose(r1, r2)
    assert not np.isclose(n1, n2)
    assert not torch.isclose(t1, t2)


def test_seed_everything_context():
    os.environ["CLINICADL_GLOBAL_SEED"] = "0"
    assert "CLINICADL_DETERMINISTIC" not in os.environ
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = "x"
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    with seed_everything_context(seed=10, deterministic=True):
        r1 = random.randint(0, 100)
        n1 = np.random.rand()
        t1 = torch.rand(1)
        assert os.environ.get("CLINICADL_GLOBAL_SEED") == "10"
        assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"
        assert os.environ.get("PYTHONHASHSEED") == "10"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert torch.are_deterministic_algorithms_enabled()

    assert os.environ.get("CLINICADL_GLOBAL_SEED") == "0"
    assert "CLINICADL_DETERMINISTIC" not in os.environ
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == "x"
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    assert not torch.are_deterministic_algorithms_enabled()

    r2 = random.randint(0, 100)
    n2 = np.random.rand()
    t2 = torch.rand(1)

    with seed_everything_context(seed=10, deterministic=True):
        r1_ = random.randint(0, 100)
        n1_ = np.random.rand()
        t1_ = torch.rand(1)

    assert np.isclose(r1, r1_)
    assert np.isclose(n1, n1_)
    assert torch.isclose(t1, t1_)

    assert not np.isclose(r1, r2)
    assert not np.isclose(n1, n2)
    assert not torch.isclose(t1, t2)


@pytest.mark.gpu
def test_seed_everything_gpu():
    with seed_everything_context(seed=10):
        assert torch.cuda.initial_seed() == 10
    assert torch.cuda.initial_seed() != 10
    seed_everything(7)
    assert torch.cuda.initial_seed() == 7


@patch("numpy.random.SeedSequence", side_effect=np.random.SeedSequence)
@patch(
    "clinicadl.utils.seed.get_rank",
    Mock(return_value=3),
)
def test_pl_worker_init_function(mock_seedsequence):
    torch.manual_seed(7)
    pl_worker_init_function(1)
    mock_seedsequence.assert_called_once_with([6, 1, 3])
