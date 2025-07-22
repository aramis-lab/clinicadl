from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

from clinicadl.callbacks.factory.lr_scheduler import LRScheduler
from clinicadl.optim.lr_schedulers.config import *

from ...resources.objects import NETWORK, OPTIMIZER, TRAINING_STATE

GOOD_PARAMETERS = [
    ({}, "ConstantLR", ConstantLRConfig, ConstantLR),
    ({"gamma": 1}, "ExponentialLR", ExponentialLRConfig, ExponentialLR),
    ({}, "LinearLR", LinearLRConfig, LinearLR),
    ({"step_size": 1}, "StepLR", StepLRConfig, StepLR),
    ({"milestones": [1, 2]}, "MultiStepLR", MultiStepLRConfig, MultiStepLR),
    ({}, "PolynomialLR", PolynomialLRConfig, PolynomialLR),
    ({}, "ReduceLROnPlateau", ReduceLROnPlateauConfig, ReduceLROnPlateau),
    ({"max_lr": 1, "total_steps": 10}, "OneCycleLR", OneCycleLRConfig, OneCycleLR),
]


@pytest.mark.parametrize(
    "args,name,config,sched",
    GOOD_PARAMETERS,
)
def test_scheduler_init(args, name, config, sched):
    scheduler_from_str = LRScheduler(name, **args)
    assert scheduler_from_str.config is not None
    assert isinstance(scheduler_from_str.config, config)

    _config = config(**args)
    scheduler_from_config = LRScheduler(_config)
    assert scheduler_from_config.config is not None
    assert scheduler_from_config.config == _config
    assert scheduler_from_config.torch_scheduler is None
    assert scheduler_from_config.scheduler is None

    scheduler_from_config.on_train_begin(TRAINING_STATE)
    assert scheduler_from_config.scheduler is not None
    assert isinstance(
        scheduler_from_config.scheduler, torch.optim.lr_scheduler.LRScheduler
    )


@pytest.mark.parametrize(
    "args,name,config,sched",
    GOOD_PARAMETERS,
)
def test_scheduler_init_with_torch_optimizer(args, name, config, sched):
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())
    torch_scheduler = sched(optimizer, **args)
    scheduler = LRScheduler(torch_scheduler)
    assert scheduler.torch_scheduler == torch_scheduler
    assert scheduler.config is None
    assert scheduler.scheduler is None

    scheduler.on_train_begin(TRAINING_STATE)
    assert scheduler.scheduler is not None
    assert isinstance(scheduler.scheduler, sched)


@pytest.mark.parametrize(
    "args,name,config,sched",
    GOOD_PARAMETERS,
)
def test_on_batch_end_steps_scheduler(args, name, config, sched):
    scheduler = LRScheduler(name, **args)
    scheduler.on_train_begin(TRAINING_STATE)

    # Mock step to verify it's called
    scheduler.scheduler.step = MagicMock()
    scheduler.on_batch_end(TRAINING_STATE)
    scheduler.scheduler.step.assert_called_once()


@pytest.mark.parametrize(
    "args,name,config,sched",
    GOOD_PARAMETERS,
)
def test_on_train_begin_raises_without_optimizer(args, name, config, sched):
    _config = deepcopy(TRAINING_STATE)
    del _config.model.optimizer  # remove optimizer

    scheduler = LRScheduler(name, **args)
    with pytest.raises(AttributeError):
        scheduler.on_train_begin(_config)

    with pytest.raises(RuntimeError):
        scheduler.on_batch_end(TRAINING_STATE)
