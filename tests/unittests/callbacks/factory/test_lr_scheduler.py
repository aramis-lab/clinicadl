from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
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
from clinicadl.optim.lr_schedulers.config import (
    ConstantLRConfig,
    ExponentialLRConfig,
    LinearLRConfig,
    MultiStepLRConfig,
    OneCycleLRConfig,
    PolynomialLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)

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
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())

    with pytest.raises(
        ValueError,
        match="If you pass a LRScheduler via a name or a config class, you must also pass the associated optimizer via 'optimizer'.",
    ):
        LRScheduler(name, **args)
    scheduler_from_str = LRScheduler(name, optimizer=optimizer, **args)
    assert scheduler_from_str.config is not None
    assert isinstance(scheduler_from_str.config, config)
    assert isinstance(scheduler_from_str.scheduler, sched)

    _config = config(**args)
    with pytest.raises(
        ValueError,
        match="If you pass a LRScheduler via a name or a config class, you must also pass the associated optimizer via 'optimizer'.",
    ):
        LRScheduler(_config, **args)
    scheduler_from_config = LRScheduler(_config, optimizer=optimizer)
    assert scheduler_from_config.config is not None
    assert scheduler_from_config.config == _config
    assert isinstance(scheduler_from_str.scheduler, sched)


def test_raw_scheduler():
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())
    raw_scheduler = ConstantLR(optimizer)
    with pytest.raises(
        ValueError,
        match="If you pass directly your own LRScheduler, you must must specify the type of scheduler via 'scheduler_type'.",
    ):
        LRScheduler(scheduler=ConstantLR(optimizer))
    scheduler_from_raw = LRScheduler(
        scheduler=raw_scheduler, scheduler_type="epoch-based"
    )
    assert scheduler_from_raw.config is None
    assert scheduler_from_raw.scheduler is raw_scheduler
    assert scheduler_from_raw.scheduler_type == "epoch-based"


def test_on_train_begin():
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())
    scheduler = LRScheduler(LinearLRConfig(start_factor=0.5), optimizer=optimizer)

    optimizer.step()
    scheduler.scheduler.step()
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006
    scheduler.on_train_begin(TRAINING_STATE)
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0005


def test_steps_scheduler():
    TRAINING_STATE.metrics._df = pd.DataFrame(
        {"epoch": [0], "mse": [1.0], "mae": [1.0], "loss": [0.5]}
    )
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())
    sched = StepLR(optimizer, step_size=1)
    epoch_scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)
    step_scheduler = LRScheduler(sched, scheduler_type="step-based")
    metric_scheduler = LRScheduler(deepcopy(sched), scheduler_type="loss-based")

    # Mock step to verify it's called
    epoch_scheduler.scheduler.step = MagicMock()
    step_scheduler.scheduler.step = MagicMock()
    metric_scheduler.scheduler.step = MagicMock()

    epoch_scheduler.on_batch_end(TRAINING_STATE)
    step_scheduler.on_batch_end(TRAINING_STATE)
    metric_scheduler.on_batch_end(TRAINING_STATE)

    epoch_scheduler.scheduler.step.assert_not_called()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_not_called()

    epoch_scheduler.on_epoch_end(TRAINING_STATE)
    step_scheduler.on_epoch_end(TRAINING_STATE)
    metric_scheduler.on_epoch_end(TRAINING_STATE)

    epoch_scheduler.scheduler.step.assert_called_once()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_called_once()


@pytest.mark.gpu
def test_save_load_checkpoint(tmp_path):
    net = NETWORK.get_object()
    optimizer = OPTIMIZER.get_object(net)
    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)

    optimizer.step()
    scheduler.scheduler.step()
    scheduler.save_checkpoint(tmp_path / "scheduler.json")

    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)
    scheduler.load_checkpoint(tmp_path / "scheduler.json")
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006

    net.to("cuda")
    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)
    scheduler.load_checkpoint(tmp_path / "scheduler.json", device=torch.device("cuda"))
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006
