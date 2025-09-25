import re
from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    StepLR,
)

from clinicadl.callbacks.factory.lr_scheduler import LRScheduler
from clinicadl.optim.lr_schedulers.config import (
    ConstantLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)

from ...resources.objects import NETWORK, OPTIMIZER, TRAINING_STATE


def test__init__():
    # raw scheduler
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

    # config
    config = ConstantLRConfig()
    scheduler_from_config = LRScheduler(config)
    assert scheduler_from_config.config is config
    assert scheduler_from_config.scheduler is None
    assert scheduler_from_config.scheduler_type == "epoch-based"

    # name
    scheduler_from_name = LRScheduler("ConstantLR")
    assert isinstance(scheduler_from_name.config, ConstantLRConfig)

    # metric
    with pytest.raises(
        ClinicaDLConfigurationError,
        match="If scheduler_type='metric-based', you must pass the name of the validation metric via 'metric_name'.",
    ):
        LRScheduler(scheduler=raw_scheduler, scheduler_type="metric-based")
    LRScheduler(
        scheduler=raw_scheduler, scheduler_type="metric-based", metric_name="mse"
    )


def test_on_train_begin():
    # raw scheduler
    optimizer = OPTIMIZER.get_object(NETWORK.get_object())
    raw_scheduler = ConstantLR(optimizer)
    scheduler = LRScheduler(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
        optimizer_name="optimizer_",
    )
    with pytest.raises(
        ClinicaDLArgumentError,
        match=(
            re.escape(
                "In LRScheduler, optimizer_name='optimizer_' but there is no such optimizer (returned by the 'get_optimizers' method of you ClinicaDLModel). "
                "Optimizers are: ['optimizer']"
            )
        ),
    ):
        scheduler.on_train_begin(TRAINING_STATE)

    scheduler = LRScheduler(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
    )
    with pytest.raises(
        ClinicaDLConfigurationError,
        match=(
            re.escape(
                "The optimizer associated to the LR scheduler 'ConstantLR' is not the same as "
                "'optimizer' (returned by the 'get_optimizers' method of you ClinicaDLModel)."
            )
        ),
    ):
        scheduler.on_train_begin(TRAINING_STATE)

    raw_scheduler = ConstantLR(TRAINING_STATE.model.optimizer)
    scheduler = LRScheduler(scheduler=raw_scheduler, scheduler_type="epoch-based")

    scheduler.on_train_begin(TRAINING_STATE)
    optimizer.step()
    scheduler.scheduler.step()
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006
    scheduler.on_train_begin(TRAINING_STATE)
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0005

    # config
    scheduler = LRScheduler(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
        optimizer_name="optimizer_",
    )
    with pytest.raises(
        ClinicaDLArgumentError,
        match=(
            re.escape(
                "In LRScheduler, optimizer_name='optimizer_' but there is no such optimizer (returned by the 'get_optimizers' method of you ClinicaDLModel). "
                "Optimizers are: ['optimizer']"
            )
        ),
    ):
        scheduler.on_train_begin(TRAINING_STATE)

    scheduler = LRScheduler(
        scheduler=raw_scheduler,
        scheduler_type="epoch-based",
    )
    scheduler.on_train_begin(TRAINING_STATE)
    assert isinstance(scheduler.scheduler, ConstantLR)
    assert scheduler.scheduler.optimizer is TRAINING_STATE.model.optimizer
    optimizer.step()
    scheduler.scheduler.step()
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006
    scheduler.on_train_begin(TRAINING_STATE)
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0005


def test_steps_scheduler():
    TRAINING_STATE.metrics._df = pd.DataFrame(
        {"epoch": [0], "mse": [1.1], "loss": [0.5]}
    )
    sched = StepLR(TRAINING_STATE.model.optimizer, step_size=1)
    epoch_scheduler = LRScheduler(StepLRConfig(step_size=1))
    step_scheduler = LRScheduler(sched, scheduler_type="step-based")
    metric_scheduler = LRScheduler(
        ReduceLROnPlateauConfig(), scheduler_type="metric-based", metric_name="mse"
    )

    epoch_scheduler.on_train_begin(TRAINING_STATE)
    step_scheduler.on_train_begin(TRAINING_STATE)
    metric_scheduler.on_train_begin(TRAINING_STATE)

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
    metric_scheduler.scheduler.step.assert_called_once_with(1.1)


@pytest.mark.gpu
def test_save_load_checkpoint(tmp_path):
    net = NETWORK.get_object()
    optimizer = OPTIMIZER.get_object(net)
    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)

    optimizer.step()
    scheduler.scheduler.step()
    scheduler.save_checkpoint(tmp_path / "scheduler")

    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)
    scheduler.load_checkpoint(tmp_path / "scheduler")
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006

    net.to("cuda")
    scheduler = LRScheduler(StepLRConfig(step_size=1), optimizer=optimizer)
    scheduler.load_checkpoint(tmp_path / "scheduler", device=torch.device("cuda"))
    scheduler.scheduler.state_dict()["_last_lr"] = 0.0006
