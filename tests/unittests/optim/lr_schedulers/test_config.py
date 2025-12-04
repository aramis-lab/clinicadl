from collections import OrderedDict

import pytest
import torch.nn as nn
import torch.optim as optim
from pydantic import ValidationError

from clinicadl.optim.lr_schedulers.config import (
    ConstantLRConfig,
    ExponentialLRConfig,
    ImplementedLRScheduler,
    LinearLRConfig,
    MultiStepLRConfig,
    OneCycleLRConfig,
    PolynomialLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
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
    optimizer = optim.SGD(
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
    return optimizer


MANDATORY_FIELDS = {
    "step_size": 1,
    "milestones": [1, 2],
    "max_lr": 1,
    "total_steps": 10,
    "gamma": 1,
}
BAD_INPUTS = [
    ({"milestones": [4, 2, 4]}, MultiStepLRConfig),
    ({"gamma": 0}, ExponentialLRConfig),
    ({"gamma": 0, "step_size": 1}, StepLRConfig),
    ({"gamma": 0, "milestones": [1, 2]}, MultiStepLRConfig),
    (
        {"last_epoch": -2, "gamma": 1},
        ExponentialLRConfig,
    ),
    (
        {"last_epoch": -2, "step_size": 1},
        StepLRConfig,
    ),
    (
        {"last_epoch": -2, "milestones": [1, 2]},
        MultiStepLRConfig,
    ),
    (
        {"last_epoch": -2, "max_lr": 1},
        OneCycleLRConfig,
    ),
    (
        {"last_epoch": -2},
        [
            ConstantLRConfig,
            LinearLRConfig,
            PolynomialLRConfig,
        ],
    ),
    ({"step_size": 0}, StepLRConfig),
    ({"factor": 0}, [ConstantLRConfig, ReduceLROnPlateauConfig]),
    ({"total_iters": 0}, [ConstantLRConfig, LinearLRConfig, PolynomialLRConfig]),
    ({"start_factor": 0}, LinearLRConfig),
    ({"end_factor": 0}, LinearLRConfig),
    ({"mode": "abc"}, ReduceLROnPlateauConfig),
    ({"patience": -1}, ReduceLROnPlateauConfig),
    ({"threshold": -1}, ReduceLROnPlateauConfig),
    ({"threshold_mode": "abc"}, ReduceLROnPlateauConfig),
    ({"cooldown": -1}, ReduceLROnPlateauConfig),
    ({"eps": -0.1}, ReduceLROnPlateauConfig),
    ({"min_lr": -0.1}, ReduceLROnPlateauConfig),
    ({"min_lr": [-0.1]}, ReduceLROnPlateauConfig),
    ({"min_lr": {"group_1": -0.1, "ELSE": 0}}, ReduceLROnPlateauConfig),
    ({"milestones": [0, 1]}, MultiStepLRConfig),
    ({"max_lr": 0, "total_steps": 10}, OneCycleLRConfig),
    ({"max_lr": [0], "total_steps": 10}, OneCycleLRConfig),
    ({"max_lr": {"group_1": 1, "ELSE": 0}, "total_steps": 10}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 0}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "pct_start": 0}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "pct_start": 1}, OneCycleLRConfig),
    ({"max_lr": 1}, OneCycleLRConfig),
    ({"max_lr": 1, "epochs": 1}, OneCycleLRConfig),
    ({"max_lr": 1, "steps_per_epoch": 1}, OneCycleLRConfig),
    ({"max_lr": 1, "epochs": 1, "steps_per_epoch": 0}, OneCycleLRConfig),
    ({"max_lr": 1, "epochs": 0, "steps_per_epoch": 1}, OneCycleLRConfig),
    (
        {"max_lr": 1, "epochs": 1, "steps_per_epoch": 1, "total_steps": 10},
        OneCycleLRConfig,
    ),
    ({"max_lr": 1, "total_steps": 10, "anneal_strategy": "abc"}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "base_momentum": -0.1}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "base_momentum": [-0.1]}, OneCycleLRConfig),
    (
        {"max_lr": 1, "total_steps": 10, "base_momentum": {"group_1": -0.1, "ELSE": 0}},
        OneCycleLRConfig,
    ),
    ({"max_lr": 1, "total_steps": 10, "max_momentum": -0.1}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "max_momentum": [-0.1]}, OneCycleLRConfig),
    (
        {"max_lr": 1, "total_steps": 10, "max_momentum": {"group_1": -0.1, "ELSE": 0}},
        OneCycleLRConfig,
    ),
    ({"max_lr": 1, "total_steps": 10, "div_factor": 0}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "final_div_factor": 0}, OneCycleLRConfig),
    ({"max_lr": 1, "total_steps": 10, "last_epoch": -2}, OneCycleLRConfig),
]

GOOD_INPUTS = [
    ({"step_size": 1, "gamma": 1, "last_epoch": -1}, StepLRConfig),
    ({"gamma": 1, "last_epoch": -1}, ExponentialLRConfig),
    ({"factor": 0.1, "total_iters": 1, "last_epoch": -1}, ConstantLRConfig),
    ({"milestones": [1, 2], "gamma": 1, "last_epoch": -1}, MultiStepLRConfig),
    (
        {"start_factor": 0.1, "end_factor": 0.2, "total_iters": 1, "last_epoch": -1},
        LinearLRConfig,
    ),
    (
        {
            "eps": 0,
            "min_lr": 0,
            "cooldown": 0,
            "threshold_mode": "abs",
            "threshold": 0,
            "patience": 0,
            "factor": 0.1,
            "mode": "min",
        },
        ReduceLROnPlateauConfig,
    ),
    (
        {
            "min_lr": [0],
        },
        ReduceLROnPlateauConfig,
    ),
    (
        {
            "min_lr": {"group_1": 1.0, "ELSE": 0.0},
        },
        ReduceLROnPlateauConfig,
    ),
    ({"power": 0, "total_iters": 1, "last_epoch": -1}, PolynomialLRConfig),
    (
        {
            "max_lr": 1,
            "total_steps": 10,
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "cycle_momentum": True,
            "max_momentum": 0,
            "base_momentum": 0,
            "div_factor": 0.5,
            "final_div_factor": 0.5,
            "three_phase": True,
            "last_epoch": -1,
        },
        OneCycleLRConfig,
    ),
    (
        {
            "max_lr": [1],
            "epochs": 1,
            "steps_per_epoch": 1,
            "anneal_strategy": "linear",
            "cycle_momentum": False,
            "three_phase": False,
            "max_momentum": [0],
            "base_momentum": [0],
        },
        OneCycleLRConfig,
    ),
    (
        {
            "max_lr": {"group_1": 1, "ELSE": 0.5},
            "total_steps": 10,
            "max_momentum": {"group_1": 1.0, "ELSE": 0.0},
            "base_momentum": {"group_1": 1.0, "ELSE": 0.0},
        },
        OneCycleLRConfig,
    ),
]


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize(
    "args,configs",
    GOOD_INPUTS,
)
def test_good_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        c = config(**args)
        for arg, value in args.items():
            assert getattr(c, arg) == value


def test_group_validator():
    with pytest.raises(ValidationError):
        ReduceLROnPlateauConfig(min_lr={"params1": 0.1})
    ReduceLROnPlateauConfig(min_lr={"params1": 0.1, "ELSE": 0.2})

    with pytest.raises(ValidationError):
        OneCycleLRConfig(
            max_lr={"params1": 0.1, "ELSE": 0.2},
            max_momentum={"params1": 0.1, "params2": 0.2, "ELSE": 0.2},
            total_steps=1,
        )
    with pytest.raises(ValidationError):
        OneCycleLRConfig(total_steps=1, max_lr={"params1": 0.1})
    with pytest.raises(ValidationError):
        OneCycleLRConfig(max_lr=1, total_steps=1, base_momentum={"params1": 0.1})
    with pytest.raises(ValidationError):
        OneCycleLRConfig(max_lr=1, total_steps=1, max_momentum={"params1": 0.1})
    OneCycleLRConfig(
        max_lr={"params1": 0.1, "ELSE": 0.2},
        base_momentum={"params1": 0.1, "ELSE": 0.2},
        max_momentum={"params1": 0.1, "ELSE": 0.2},
        total_steps=1,
    )


@pytest.mark.parametrize(
    "args,config,expected_class",
    [
        ({}, ConstantLRConfig, optim.lr_scheduler.ConstantLR),
        ({"gamma": 1}, ExponentialLRConfig, optim.lr_scheduler.ExponentialLR),
        ({}, LinearLRConfig, optim.lr_scheduler.LinearLR),
        ({"milestones": [1, 2]}, MultiStepLRConfig, optim.lr_scheduler.MultiStepLR),
        (
            {"max_lr": 1, "total_steps": 10},
            OneCycleLRConfig,
            optim.lr_scheduler.OneCycleLR,
        ),
        ({}, PolynomialLRConfig, optim.lr_scheduler.PolynomialLR),
        ({}, ReduceLROnPlateauConfig, optim.lr_scheduler.ReduceLROnPlateau),
        ({"step_size": 1}, StepLRConfig, optim.lr_scheduler.StepLR),
    ],
)
def test_get_object(args, config, expected_class, optimizer, network):
    c = config(**args)
    scheduler = c.get_object(optimizer)
    assert isinstance(scheduler, expected_class)

    if c.name == "OneCycleLR":
        with pytest.raises(
            ValueError,
            match=r"^There are 3 parameter groups in the optimizer, but 2 groups in the OneCycleLR for parameter 'max_lr'. Make sure that the parameter groups match between your optimizer and LR scheduler!$",
        ):
            OneCycleLRConfig(
                max_lr={"linear2": 0.01, "ELSE": 10},
                total_steps=1,
                base_momentum=0.33,
                cycle_momentum=True,
            ).get_object(optimizer)

        with pytest.raises(
            ValueError,
            match=r"^There are 3 parameter groups in the optimizer, but 2 groups in the OneCycleLR for parameter 'max_momentum'. Make sure that the parameter groups match between your optimizer and LR scheduler!$",
        ):
            OneCycleLRConfig(
                max_lr=1,
                max_momentum=[0.01, 0],
                total_steps=1,
                base_momentum=0.33,
                cycle_momentum=True,
            ).get_object(optimizer)

        with pytest.raises(
            ValueError,
            match="If 'cycle_momentum' is True in OneCycleLR, the optimizer requires a momentum.",
        ):
            OneCycleLRConfig(
                max_lr=1,
                total_steps=1,
                base_momentum=0.33,
                cycle_momentum=True,
            ).get_object(optim.Adagrad(network.parameters()))

    if c.name == "ReduceLROnPlateau":
        # check consistency between optimizer and lr scheduler configs
        from clinicadl.optim.optimizers.config import AdamConfig

        optimizer = AdamConfig(
            lr={"linear2": 0.01, "linear1": 0.1, "ELSE": 1}
        ).get_object(network)

        scheduler: optim.lr_scheduler.ReduceLROnPlateau = ReduceLROnPlateauConfig(
            min_lr={"linear2": 0.01, "ELSE": 0, "linear1": 0.1},
        ).get_object(optimizer)

        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[1]["lr"] == 0.01
        assert optimizer.param_groups[2]["lr"] == 1

        assert scheduler.min_lrs[0] == 0.1
        assert scheduler.min_lrs[1] == 0.01
        assert scheduler.min_lrs[2] == 0


def test_name():
    for name in ImplementedLRScheduler:
        config = globals()[f"{name.value}Config"]
    c = config(**MANDATORY_FIELDS)
    assert c.name == name.value


@pytest.mark.parametrize(
    "config,type_",
    [
        (ConstantLRConfig, "epoch-based"),
        (ExponentialLRConfig, "epoch-based"),
        (LinearLRConfig, "epoch-based"),
        (StepLRConfig, "epoch-based"),
        (MultiStepLRConfig, "epoch-based"),
        (PolynomialLRConfig, "epoch-based"),
        (ReduceLROnPlateauConfig, "metric-based"),
        (
            OneCycleLRConfig,
            "step-based",
        ),
    ],
)
def test_scheduler_type(config, type_):
    assert config.scheduler_type() == type_
