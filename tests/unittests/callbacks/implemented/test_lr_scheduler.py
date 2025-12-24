import re
from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.optim.lr_scheduler import (
    ConstantLR,
    StepLR,
)

from clinicadl.callbacks import LRSchedulerCallback
from clinicadl.optim.lr_schedulers.config import (
    ConstantLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)
from clinicadl.train import TrainerState
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)


def build_optimizer(key="optimizer"):
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters())

    return {key: optimizer}


def test__init__():
    # raw scheduler
    optimizer = build_optimizer()["optimizer"]
    raw_scheduler = ConstantLR(optimizer)
    with pytest.raises(
        ValidationError,
        match="If you pass directly your own LRScheduler, you must specify the type of scheduler via 'scheduler_type'",
    ):
        LRSchedulerCallback(scheduler=ConstantLR(optimizer))
    scheduler_from_raw = LRSchedulerCallback(
        scheduler=raw_scheduler, scheduler_type="epoch-based"
    )
    assert scheduler_from_raw.scheduler_config is None
    assert scheduler_from_raw.scheduler is raw_scheduler

    # config
    config = ConstantLRConfig()
    scheduler_from_config = LRSchedulerCallback(config)
    assert scheduler_from_config.scheduler_config is config
    assert scheduler_from_config.scheduler is None
    assert scheduler_from_config.config.scheduler_type == "epoch-based"

    # metric
    with pytest.raises(
        ValidationError,
        match="If scheduler_type='metric-based', you must pass the name of the validation metric via 'metric_name'.",
    ):
        LRSchedulerCallback(scheduler=raw_scheduler, scheduler_type="metric-based")
    LRSchedulerCallback(
        scheduler=raw_scheduler, scheduler_type="metric-based", metric_name="mse"
    )


def test_on_train_start():
    # raw scheduler
    optimizer = build_optimizer("my_optimizer")
    raw_scheduler = StepLR(optimizer["my_optimizer"], step_size=1)
    scheduler = LRSchedulerCallback(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
    )
    with pytest.raises(
        ClinicaDLArgumentError,
        match=(
            re.escape(
                "In LRSchedulerCallback, optimizer_name='optimizer' but there is no such optimizer (built with 'build_optimizers' method of your clinicadl.model.Model). "
                "Optimizers are: ['my_optimizer']"
            )
        ),
    ):
        scheduler.on_train_start(optimizers=optimizer)

    scheduler = LRSchedulerCallback(
        scheduler=deepcopy(raw_scheduler),
        scheduler_type="epoch-based",
        optimizer_name="my_optimizer",
    )
    with pytest.raises(
        ClinicaDLConfigurationError,
        match=(
            re.escape(
                "The optimizer associated to the LR scheduler StepLR is not the same as "
                "'my_optimizer' (returned by 'build_optimizers' method of your clinicadl.model.Model)."
            )
        ),
    ):
        scheduler.on_train_start(optimizers=optimizer)

    scheduler = LRSchedulerCallback(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
        optimizer_name="my_optimizer",
    )
    scheduler.on_train_start(optimizers=optimizer)
    optimizer["my_optimizer"].step()
    scheduler.scheduler.step()
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.0001]

    scheduler.on_train_start(optimizers=optimizer)
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.001]

    # config
    optimizer = build_optimizer()
    scheduler = LRSchedulerCallback(
        scheduler=StepLRConfig(step_size=1),
    )
    scheduler.on_train_start(optimizers=optimizer)
    assert isinstance(scheduler.scheduler, StepLR)
    assert scheduler.scheduler.optimizer is optimizer["optimizer"]
    optimizer["optimizer"].step()
    scheduler.scheduler.step()
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.0001]

    optimizer = build_optimizer()
    scheduler.on_train_start(optimizers=optimizer)
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.001]


def test_steps_scheduler():
    optimizer = build_optimizer()
    wrong_metrics = pd.DataFrame({"epoch": [0, 1], "loss": [0.1, 0.5]})
    metrics = pd.DataFrame({"epoch": [0, 1], "mse": [0.7, 1.1], "loss": [0.1, 0.5]})
    sched = StepLR(optimizer["optimizer"], step_size=1)
    epoch_scheduler = LRSchedulerCallback(StepLRConfig(step_size=1))
    step_scheduler = LRSchedulerCallback(sched, scheduler_type="step-based")
    metric_scheduler = LRSchedulerCallback(
        ReduceLROnPlateauConfig(), scheduler_type="metric-based", metric_name="mse"
    )

    epoch_scheduler.on_train_start(optimizers=optimizer)
    step_scheduler.on_train_start(optimizers=optimizer)
    metric_scheduler.on_train_start(optimizers=optimizer)

    epoch_scheduler.scheduler.step = MagicMock()
    step_scheduler.scheduler.step = MagicMock()
    metric_scheduler.scheduler.step = MagicMock()

    epoch_scheduler.on_optimization_step_end(optimizers=optimizer)
    step_scheduler.on_optimization_step_end(optimizers=optimizer)
    metric_scheduler.on_optimization_step_end(optimizers=optimizer)

    epoch_scheduler.scheduler.step.assert_not_called()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_not_called()

    epoch_scheduler.on_epoch_end()
    step_scheduler.on_epoch_end()
    metric_scheduler.on_epoch_end()

    epoch_scheduler.scheduler.step.assert_called_once()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_not_called()

    state = TrainerState(current_epoch=1)
    epoch_scheduler.on_validation_end(state=state, metrics_df=metrics)
    step_scheduler.on_validation_end(state=state, metrics_df=metrics)
    metric_scheduler.on_validation_end(state=state, metrics_df=metrics)

    epoch_scheduler.scheduler.step.assert_called_once()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_called_once_with(1.1)

    with pytest.raises(KeyError, match="'mse' not found in the validation metrics!"):
        metric_scheduler.on_validation_end(state=state, metrics_df=wrong_metrics)

    # only validation
    metric_scheduler = LRSchedulerCallback(
        ReduceLROnPlateauConfig(), scheduler_type="metric-based", metric_name="mse"
    )
    metric_scheduler.scheduler = MagicMock()
    metric_scheduler.on_validation_end(state=state, metrics_df=metrics)
    metric_scheduler.scheduler.step.assert_not_called()


def test_from_dict_to_dict():
    scheduler = LRSchedulerCallback(
        scheduler=StepLRConfig(step_size=3),
        optimizer_name="my_optimizer",
    )
    assert isinstance(
        new_scheduler := LRSchedulerCallback.from_dict(scheduler.to_dict()),
        LRSchedulerCallback,
    )
    assert new_scheduler.config.optimizer_name == "my_optimizer"
    assert new_scheduler.config.scheduler.value.step_size == 3

    optimizer = build_optimizer()
    scheduler = StepLR(optimizer["optimizer"], step_size=3)
    scheduler = LRSchedulerCallback(
        scheduler=scheduler,
        scheduler_type="metric-based",
        metric_name="abc",
    )
    new_scheduler = LRSchedulerCallback.from_dict(scheduler.to_dict())
    assert new_scheduler.config.scheduler_type == "metric-based"
    assert new_scheduler.config.metric_name == "abc"


def test_state_dict():
    optimizer = build_optimizer()
    scheduler = LRSchedulerCallback(
        scheduler=StepLRConfig(step_size=1),
    )
    scheduler.on_train_start(optimizers=optimizer)
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.001]
    optimizer["optimizer"].step()
    scheduler.scheduler.step()
    state_dict = scheduler.state_dict()
    optimizer["optimizer"].step()
    scheduler.scheduler.step()
    assert scheduler.scheduler.last_epoch == 2
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.00001]
    scheduler.load_state_dict(state_dict)
    assert scheduler.scheduler.last_epoch == 1
    assert scheduler.scheduler.state_dict()["_last_lr"] == [0.0001]
