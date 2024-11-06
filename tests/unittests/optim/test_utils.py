from collections import OrderedDict

import numpy as np
import pytest
import torch.nn as nn

from clinicadl.optim import (
    get_lr_scheduler_config,
    get_lr_scheduler_from_config,
    get_optimizer_config,
    get_optimizer_from_config,
)
from clinicadl.optim.utils import check_optimizer_scheduler_consistency


@pytest.mark.parametrize(
    "optimizer_args,scheduler_args,error",
    [
        (
            {"name": "SGD", "momentum": 1},
            {"name": "ReduceLROnPlateau", "min_lr": 0},
            False,
        ),
        (
            {"name": "SGD", "momentum": {"linear1": 2, "ELSE": 1}},
            {"name": "ReduceLROnPlateau", "min_lr": 0},
            False,
        ),
        (
            {"name": "SGD", "momentum": {"linear1": 2, "ELSE": 1}},
            {"name": "StepLR", "step_size": 1},
            False,
        ),
        (
            {"name": "SGD", "momentum": 1},
            {"name": "ReduceLROnPlateau", "min_lr": {"linear1": 0.1, "ELSE": 0}},
            True,
        ),
        (
            {"name": "SGD", "momentum": {"linear1": 2, "ELSE": 1}},
            {"name": "ReduceLROnPlateau", "min_lr": {"linear1": 0.1, "ELSE": 0.33}},
            False,
        ),
        (
            {
                "name": "SGD",
                "momentum": {"linear1": 2, "ELSE": 1},
                "dampening": {"linear2": 1, "ELSE": 1},
            },
            {"name": "ReduceLROnPlateau", "min_lr": {"linear1": 0.1, "ELSE": 0.33}},
            True,
        ),
        (
            {
                "name": "SGD",
                "momentum": {"linear1": 2, "ELSE": 1},
                "dampening": {"linear2": 1, "ELSE": 0},
            },
            {
                "name": "ReduceLROnPlateau",
                "min_lr": {"linear1": 0.1, "linear2": 0.2, "ELSE": 0.33},
            },
            False,
        ),
    ],
)
def test_check_optimizer_scheduler_consistency(optimizer_args, scheduler_args, error):
    optimizer_config = get_optimizer_config(**optimizer_args)
    lr_scheduler_config = get_lr_scheduler_config(**scheduler_args)

    if error:
        with pytest.raises(ValueError):
            check_optimizer_scheduler_consistency(optimizer_config, lr_scheduler_config)
    else:
        check_optimizer_scheduler_consistency(optimizer_config, lr_scheduler_config)


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


def test_param_groups(network):
    optimizer_config = get_optimizer_config(
        "SGD",
        lr=1,
        weight_decay={"linear1": 0.1, "ELSE": 0},
        dampening={"linear2": 1, "ELSE": 0},
    )
    scheduler_config = get_lr_scheduler_config(
        "ReduceLROnPlateau",
        factor=0.1,
        patience=0,
        min_lr={"linear1": 0.01, "linear2": 0.1, "ELSE": 0},
    )
    check_optimizer_scheduler_consistency(optimizer_config, scheduler_config)

    optimizer, _ = get_optimizer_from_config(optimizer_config, network)
    scheduler, _ = get_lr_scheduler_from_config(scheduler_config, optimizer)
    scheduler.step(1)
    scheduler.step(1)
    scheduler.step(1)
    scheduler.step(1)
    assert np.isclose(scheduler.get_last_lr(), [0.01, 0.1, 0.001], rtol=1e-5).all()
