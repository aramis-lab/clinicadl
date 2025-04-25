from collections import OrderedDict

import numpy as np
import pytest
import torch.nn as nn

from clinicadl.optim.lr_schedulers.config import get_lr_scheduler_config
from clinicadl.optim.optimizers.config import get_optimizer_config
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
                "dampening": {"linear2": 1, "ELSE": 1},
            },
            {
                "name": "OneCycleLR",
                "max_lr": {"linear1": 0.1, "ELSE": 0.33},
                "total_steps": 10,
            },
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
        (
            {"name": "Adadelta"},
            {"name": "OneCycleLR", "max_lr": 1, "total_steps": 10},
            True,
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
        lr={"linear1": 0.1, "ELSE": 0.01},
        momentum={"linear2": 0.1, "ELSE": 0.01},
    )

    optimizer = optimizer_config.get_object(network)
    scheduler_config = get_lr_scheduler_config(
        "ReduceLROnPlateau",
        factor=0.1,
        patience=0,
        min_lr={"linear1": 0.01, "linear2": 0.001, "ELSE": 0},
    )
    scheduler = scheduler_config.get_object(optimizer)
    optimizer.step()
    scheduler.step(1)
    scheduler.step(1)
    scheduler.step(1)
    assert np.isclose(optimizer.param_groups[0]["lr"], 0.01)
    assert np.isclose(optimizer.param_groups[1]["lr"], 0.001)
    assert np.isclose(optimizer.param_groups[2]["lr"], 0.0001)
    assert np.isclose(optimizer.param_groups[0]["momentum"], 0.01)
    assert np.isclose(optimizer.param_groups[1]["momentum"], 0.1)
    assert np.isclose(optimizer.param_groups[2]["momentum"], 0.01)

    optimizer = optimizer_config.get_object(network)
    scheduler_config = get_lr_scheduler_config(
        "OneCycleLR",
        total_steps=10,
        max_lr={"linear1": 0.1, "linear2": 0.01, "ELSE": 0.01},
        base_momentum={"linear2": 0.1, "linear1": 0.01, "ELSE": 0.01},
        max_momentum=10,
    )
    scheduler = scheduler_config.get_object(optimizer)
    optimizer.step()
    scheduler.step()
    scheduler.step()
    scheduler.step()
    assert np.isclose(optimizer.param_groups[0]["lr"], 0.09504846320134738)
    assert np.isclose(optimizer.param_groups[1]["lr"], 0.009504846320134737)
    assert np.isclose(optimizer.param_groups[2]["lr"], 0.009504846320134737)
    assert np.isclose(optimizer.param_groups[0]["momentum"], 0.5046605048274166)
    assert np.isclose(optimizer.param_groups[1]["momentum"], 0.5902041038830248)
    assert np.isclose(optimizer.param_groups[2]["momentum"], 0.5046605048274166)
