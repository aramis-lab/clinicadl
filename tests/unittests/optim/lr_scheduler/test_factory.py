from collections import OrderedDict

import pytest
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from clinicadl.optim.lr_scheduler.config import (
    ImplementedLRScheduler,
    create_lr_scheduler_config,
)
from clinicadl.optim.lr_scheduler.factory import (
    get_lr_scheduler_config,
    get_lr_scheduler_from_config,
)


@pytest.fixture
def network():
    net = nn.Sequential(
        OrderedDict(
            [
                ("linear1", nn.Linear(4, 3)),
                ("linear2", nn.Linear(3, 2)),
                ("linear3", nn.Linear(2, 1)),
            ]
        )
    )
    return net


@pytest.fixture
def optimizer(network):
    optim = SGD(
        [
            {
                "params": network.linear1.parameters(),
                "lr": 1.0,
            },
            {
                "params": network.linear2.parameters(),
            },
            {
                "params": network.linear3.parameters(),
            },
        ],
        lr=10.0,
    )
    return optim


def test_get_lr_scheduler_from_config(optimizer):
    # test all lr schedulers
    args = {"step_size": 1, "milestones": [1, 2]}
    for scheduler in ImplementedLRScheduler:
        config = create_lr_scheduler_config(scheduler=scheduler)(**args)
        scheduler, _ = get_lr_scheduler_from_config(config, optimizer=optimizer)

    # test arguments
    config = create_lr_scheduler_config(scheduler="ReduceLROnPlateau")(
        mode="max",
        factor=0.123,
        threshold=1e-1,
        cooldown=3,
        min_lr={"linear2": 0.01, "linear1": 0.1, "ELSE": 0},
    )
    scheduler, updated_config = get_lr_scheduler_from_config(
        config, optimizer=optimizer
    )
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.123
    assert scheduler.patience == 10
    assert scheduler.threshold == 1e-1
    assert scheduler.threshold_mode == "rel"
    assert scheduler.cooldown == 3
    assert scheduler.min_lrs == [0.1, 0.01, 0.0]
    assert scheduler.eps == 1e-8

    assert updated_config.name == "ReduceLROnPlateau"
    assert updated_config.mode == "max"
    assert updated_config.factor == 0.123
    assert updated_config.patience == 10
    assert updated_config.threshold == 1e-1
    assert updated_config.threshold_mode == "rel"
    assert updated_config.cooldown == 3
    assert updated_config.min_lr == {"linear2": 0.01, "linear1": 0.1, "ELSE": 0}
    assert updated_config.eps == 1e-8

    config.min_lr = 1
    scheduler, updated_config = get_lr_scheduler_from_config(
        config, optimizer=optimizer
    )
    assert scheduler.min_lrs == [1.0, 1.0, 1.0]

    # no lr scheduler
    scheduler, updated_config = get_lr_scheduler_from_config(None, optimizer=optimizer)
    assert isinstance(scheduler, LambdaLR)
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [1.0, 10.0, 10.0]


def test_get_optimizer_config():
    config = get_lr_scheduler_config(
        "StepLR",
        step_size=1,
    )
    assert config.name == "StepLR"
    assert config.step_size == 1
    assert config.gamma == 0.1

    with pytest.raises(ValueError):
        get_lr_scheduler_config("abc", step_size=1)
