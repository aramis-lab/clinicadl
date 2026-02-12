import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.optim.lr_scheduler import (
    ConstantLR,
    StepLR,
)

from clinicadl.callbacks import LRSchedulerCallback
from clinicadl.io import Maps
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import MSEMetricConfig
from clinicadl.optim.lr_schedulers.config import (
    ConstantLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)
from clinicadl.train import TrainerState

MSE = MSEMetricConfig().get_object()
MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
METRICS_HANDLER = MetricsHandler()


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


def test_on_train_start(caplog):
    METRICS_HANDLER.config.metrics = {"mae": MSE}
    METRICS_HANDLER.init_metrics()

    # raw scheduler
    optimizer = build_optimizer("my_optimizer")
    raw_scheduler = StepLR(optimizer["my_optimizer"], step_size=1)
    scheduler = LRSchedulerCallback(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
    )
    with pytest.raises(
        KeyError,
        match=(
            re.escape(
                "In LRSchedulerCallback, optimizer_name='optimizer' but there is no such optimizer (built with 'build_optimizers' method of your clinicadl.model.Model). "
                "Optimizers are: ['my_optimizer']"
            )
        ),
    ):
        scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)

    scheduler = LRSchedulerCallback(
        scheduler=deepcopy(raw_scheduler),
        scheduler_type="epoch-based",
        optimizer_name="my_optimizer",
    )
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "The optimizer associated to the LR scheduler StepLR is not the same as "
                "'my_optimizer' (returned by 'build_optimizers' method of your clinicadl.model.Model)."
            )
        ),
    ):
        scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)

    scheduler = LRSchedulerCallback(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
        optimizer_name="my_optimizer",
    )
    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    optimizer["my_optimizer"].step()
    scheduler.scheduler.step()
    np.testing.assert_almost_equal(
        scheduler.scheduler.state_dict()["_last_lr"], [0.0001]
    )

    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    np.testing.assert_almost_equal(
        scheduler.scheduler.state_dict()["_last_lr"], [0.001]
    )

    # config
    optimizer = build_optimizer()
    scheduler = LRSchedulerCallback(
        scheduler=StepLRConfig(step_size=1),
    )
    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    assert isinstance(scheduler.scheduler, StepLR)
    assert scheduler.scheduler.optimizer is optimizer["optimizer"]
    optimizer["optimizer"].step()
    scheduler.scheduler.step()
    np.testing.assert_almost_equal(
        scheduler.scheduler.state_dict()["_last_lr"], [0.0001]
    )

    optimizer = build_optimizer()
    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    np.testing.assert_almost_equal(
        scheduler.scheduler.state_dict()["_last_lr"], [0.001]
    )

    # ReduceLROnPlateau
    metric_scheduler = LRSchedulerCallback(
        ReduceLROnPlateauConfig(mode="max"),
        scheduler_type="metric-based",
        metric_name="mse",
    )

    with pytest.raises(
        KeyError,
        match=re.escape(
            "'mse' not found in the computed metrics! Metrics are: ['mae']"
        ),
    ):
        METRICS_HANDLER.config.metrics = {"mae": MSE}
        metric_scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    METRICS_HANDLER.config.metrics = {"mse": MSE}
    METRICS_HANDLER.init_metrics()
    with caplog.at_level(logging.WARNING):
        metric_scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    assert (
        "Found mode='max' in ReduceLROnPlateau, but found optimum='min' in 'mse'. This may be an error."
        in caplog.text
    )


def test_steps_scheduler():
    optimizer = build_optimizer()
    state = TrainerState(called="train")
    METRICS_HANDLER.config.metrics = {"mse": MSE}
    METRICS_HANDLER.init_metrics()
    METRICS_HANDLER._df = pd.DataFrame(
        {"epoch": [0, 1], "mse": [0.7, 1.1], "loss": [0.1, 0.5]}
    )

    sched = StepLR(optimizer["optimizer"], step_size=1)
    epoch_scheduler = LRSchedulerCallback(StepLRConfig(step_size=1))
    step_scheduler = LRSchedulerCallback(sched, scheduler_type="step-based")
    metric_scheduler = LRSchedulerCallback(
        ReduceLROnPlateauConfig(), scheduler_type="metric-based", metric_name="mse"
    )

    epoch_scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    step_scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    metric_scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)

    epoch_scheduler.scheduler.step = MagicMock()
    step_scheduler.scheduler.step = MagicMock()
    metric_scheduler.scheduler.step = MagicMock()

    epoch_scheduler.on_optimization_step_end(optimizers=optimizer, state=state)
    step_scheduler.on_optimization_step_end(optimizers=optimizer, state=state)
    metric_scheduler.on_optimization_step_end(optimizers=optimizer, state=state)

    epoch_scheduler.scheduler.step.assert_not_called()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_not_called()

    epoch_scheduler.on_epoch_end(state=state)
    step_scheduler.on_epoch_end(state=state)
    metric_scheduler.on_epoch_end(state=state)

    epoch_scheduler.scheduler.step.assert_called_once()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_not_called()

    state = TrainerState(current_epoch=1, called="train")
    epoch_scheduler.on_validation_end(state=state, metrics=METRICS_HANDLER)
    step_scheduler.on_validation_end(state=state, metrics=METRICS_HANDLER)
    metric_scheduler.on_validation_end(state=state, metrics=METRICS_HANDLER)

    epoch_scheduler.scheduler.step.assert_called_once()
    step_scheduler.scheduler.step.assert_called_once()
    metric_scheduler.scheduler.step.assert_called_once_with(1.1)


def test_on_train_end(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.training.create_split(2)
    optimizer = torch.optim.SGD(
        [
            {
                "params": torch.nn.Linear(1, 1).parameters(),
                "lr": 0.1,
                "name": "param_1",
            },
            {
                "params": torch.nn.Linear(1, 1).parameters(),
                "lr": 0.01,
                "name": "param_2",
            },
        ]
    )
    scheduler = LRSchedulerCallback(StepLRConfig(step_size=1), optimizer_name="my_opt")
    scheduler.on_train_start(optimizers={"my_opt": optimizer}, metrics=METRICS_HANDLER)

    state = TrainerState(
        current_epoch=1,
        current_train_batch=4,
        split_idx=2,
        num_epochs=3,
        num_train_batches=4,
    )
    scheduler.on_epoch_end(state=state)

    state.current_epoch = 2
    state.current_train_batch = 2
    scheduler.on_epoch_end(state=state)

    state.current_epoch = 3
    state.current_train_batch = 4
    scheduler.on_train_end(maps=maps, state=state)
    df = maps.open_file(
        maps.training.splits[state.split_idx].logs.learning_rates / "my_opt.tsv"
    )
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "epoch": [1] * 4 + [2] * 4 + [3] * 4,
                "batch": [1, 2, 3, 4] * 3,
                "param_1": [0.1] * 4 + [0.01] * 2 + [0.001] * 6,
                "param_2": [0.01] * 4 + [0.001] * 2 + [0.0001] * 6,
            }
        ),
    )


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
    state = TrainerState(
        current_epoch=1,
        current_train_batch=1,
    )

    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    np.testing.assert_almost_equal(scheduler.scheduler.state_dict()["_last_lr"], [1e-3])
    optimizer["optimizer"].step()
    scheduler.on_epoch_end(state=state)
    state.current_epoch = 2
    optimizer["optimizer"].step()
    scheduler.on_epoch_end(state=state)
    state_dict = scheduler.state_dict()

    state.current_epoch = 3
    optimizer["optimizer"].step()
    scheduler.on_epoch_end(optimizers=optimizer, state=state)
    assert scheduler.scheduler.last_epoch == 3
    np.testing.assert_almost_equal(scheduler.scheduler.state_dict()["_last_lr"], [1e-6])

    scheduler.load_state_dict(state_dict)
    assert scheduler.scheduler.last_epoch == 2
    np.testing.assert_almost_equal(scheduler.scheduler.state_dict()["_last_lr"], [1e-5])
    assert scheduler._lrs == {(1, 1): [1e-3], (2, 1): [1e-4]}
    np.testing.assert_almost_equal(scheduler._current_lrs, [1e-5])


def test_resume():
    optimizer = build_optimizer()
    scheduler = LRSchedulerCallback(
        scheduler=StepLRConfig(step_size=1),
    )
    scheduler.on_train_start(optimizers=optimizer, metrics=METRICS_HANDLER)
    state_dict = scheduler.state_dict()

    new_scheduler = LRSchedulerCallback.from_dict(scheduler.to_dict())
    new_scheduler.on_resume(optimizers=optimizer)
    new_scheduler.load_state_dict(state_dict)
