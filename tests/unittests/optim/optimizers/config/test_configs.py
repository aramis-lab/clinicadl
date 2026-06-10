from collections import OrderedDict

import pytest
import torch.nn as nn
import torch.optim as optim
from pydantic import ValidationError

from clinicadl.optim.optimizers.config import (
    AdadeltaConfig,
    AdagradConfig,
    AdamConfig,
    ImplementedOptimizer,
    RMSpropConfig,
    SGDConfig,
)

BAD_INPUTS = [
    ({"lr": 0}, [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig]),
    ({"rho": 1.1}, AdadeltaConfig),
    ({"eps": -0.1}, [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig]),
    (
        {"weight_decay": -0.1},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    ({"lr_decay": -0.1}, AdagradConfig),
    ({"initial_accumulator_value": -0.1}, AdagradConfig),
    ({"betas": (0.9, 1.1)}, AdamConfig),
    ({"alpha": -0.1}, RMSpropConfig),
    ({"momentum": -0.1}, [RMSpropConfig, SGDConfig]),
]

GOOD_INPUTS = [
    (
        {"lr": 0.1},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    ({"rho": 0}, AdadeltaConfig),
    ({"eps": 0}, [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig]),
    (
        {"weight_decay": 0},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    (
        {"foreach": None},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    ({"capturable": False}, [AdadeltaConfig, AdamConfig, RMSpropConfig]),
    (
        {"maximize": True},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    (
        {"differentiable": False},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    ({"fused": None}, [AdagradConfig, AdamConfig, SGDConfig]),
    ({"lr_decay": 0}, AdagradConfig),
    ({"initial_accumulator_value": 0}, AdagradConfig),
    ({"betas": (0.0, 0.0)}, AdamConfig),
    ({"amsgrad": True}, AdamConfig),
    ({"alpha": 10}, RMSpropConfig),
    ({"momentum": 10}, [RMSpropConfig, SGDConfig]),
    ({"centered": True}, RMSpropConfig),
    ({"dampening": -1}, SGDConfig),
    ({"nesterov": True}, SGDConfig),
    (
        {"freeze": "params1"},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    (
        {"foreach": True},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
    ({"fused": False}, [AdagradConfig, AdamConfig, SGDConfig]),
    (
        {"freeze": ["params1", "params2"]},
        [AdadeltaConfig, AdagradConfig, AdamConfig, RMSpropConfig, SGDConfig],
    ),
]


@pytest.fixture
def network():
    net = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 1, kernel_size=3)),
                ("dense1", nn.Linear(10, 10)),
            ]
        )
    )
    net.add_module(
        "final",
        nn.Sequential(
            OrderedDict([("dense2", nn.Linear(10, 5)), ("dense3", nn.Linear(5, 3))])
        ),
    )
    return net


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)

        # test dict inputs
        args_dict = {
            arg: {"group_1": value, "ELSE": value}
            for arg, value in args.items()
            if arg != "freeze"
        }
        with pytest.raises(ValidationError):
            config(**args_dict)


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
            if arg == "freeze":
                assert getattr(c, arg) == value if isinstance(value, list) else [value]
            else:
                assert getattr(c, arg) == value

        # test dict inputs
        args_dict = {
            arg: {"group_1": value, "ELSE": value}
            for arg, value in args.items()
            if arg != "freeze"
        }
        c = config(**args_dict)
        for arg, value in args.items():
            if arg != "freeze":
                assert getattr(c, arg) == {"group_1": value, "ELSE": value}


def test_get_check_else():
    with pytest.raises(ValidationError):
        SGDConfig(lr={"params1": 0.1, "ELSE": 0.2}, nesterov={"params2": False})
    SGDConfig(
        lr={"params1": 0.1, "ELSE": 0.2}, nesterov={"params2": False, "ELSE": True}
    )


def test_check_param_groups():
    with pytest.raises(ValidationError):
        SGDConfig(
            lr={"params1": 0.1, "ELSE": 0.2},
            nesterov={"params2": False, "ELSE": True},
            freeze="params2",
        )
    SGDConfig(
        lr={"params1": 0.1, "ELSE": 0.2},
        nesterov={"params2": False, "ELSE": True},
        freeze="params3",
    )


@pytest.mark.parametrize(
    "config,expected_class",
    [
        (AdadeltaConfig, optim.Adadelta),
        (AdagradConfig, optim.Adagrad),
        (AdamConfig, optim.Adam),
        (RMSpropConfig, optim.RMSprop),
        (SGDConfig, optim.SGD),
    ],
)
def test_get_object(config, expected_class, network):
    c = config()
    optimizer = c.get_object(network)
    assert isinstance(optimizer, expected_class)
    assert len(optimizer.param_groups) == 1

    if c.name_ == "Adagrad":
        # test arguments
        c = AdagradConfig(
            lr=1e-5,
            weight_decay={"final.dense3.weight": 1.0, "dense1": 0.1, "ELSE": 0.2},
            lr_decay={"final.dense3.bias": 1, "dense1": 10, "ELSE": 100},
            eps={"ELSE": 1.0},
        )
        optimizer = c.get_object(network)
        assert len(optimizer.param_groups) == 4

        assert len(optimizer.param_groups[0]["params"]) == 2
        assert len(optimizer.param_groups[1]["params"]) == 1
        assert len(optimizer.param_groups[2]["params"]) == 1
        assert len(optimizer.param_groups[3]["params"]) == 4

        assert optimizer.param_groups[0]["lr"] == 1e-5
        assert optimizer.param_groups[1]["lr"] == 1e-5
        assert optimizer.param_groups[2]["lr"] == 1e-5
        assert optimizer.param_groups[3]["lr"] == 1e-5

        assert optimizer.param_groups[0]["lr_decay"] == 10
        assert optimizer.param_groups[1]["lr_decay"] == 1
        assert optimizer.param_groups[2]["lr_decay"] == 100
        assert optimizer.param_groups[3]["lr_decay"] == 100

        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.2
        assert optimizer.param_groups[2]["weight_decay"] == 1.0
        assert optimizer.param_groups[3]["weight_decay"] == 0.2

        assert optimizer.param_groups[0]["eps"] == 1.0
        assert optimizer.param_groups[1]["eps"] == 1.0
        assert optimizer.param_groups[2]["eps"] == 1.0
        assert optimizer.param_groups[3]["eps"] == 1.0

        assert not optimizer.param_groups[0]["differentiable"]
        assert not optimizer.param_groups[1]["differentiable"]
        assert not optimizer.param_groups[2]["differentiable"]
        assert not optimizer.param_groups[3]["differentiable"]

        assert optimizer.param_groups[0]["name"] == "dense1"
        assert optimizer.param_groups[1]["name"] == "final.dense3.bias"
        assert optimizer.param_groups[2]["name"] == "final.dense3.weight"
        assert optimizer.param_groups[3]["name"] == "ELSE"

        # special case
        c = AdagradConfig(
            lr_decay={"ELSE": 100},
        )
        optimizer = c.get_object(network)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr_decay"] == 100

        # test freeze
        c = AdagradConfig(lr_decay={"final": 10, "ELSE": 100}, freeze="conv1")
        optimizer = c.get_object(network)
        assert len(optimizer.param_groups) == 2
        for param in network.conv1.parameters():
            assert not param.requires_grad
        for param in network.final.parameters():
            assert param.requires_grad
        for param in network.dense1.parameters():
            assert param.requires_grad


def test_get_parameters_group():
    optimizer = SGDConfig(
        weight_decay={"param_1": 0, "param_0": 0, "ELSE": 0},
        momentum={"param_1": 0, "param_3.linear": 0, "ELSE": 0},
    )
    assert optimizer.get_parameter_groups() == [
        "param_0",
        "param_1",
        "param_3.linear",
        "ELSE",
    ]

    optimizer = SGDConfig(
        weight_decay=0,
        momentum=0,
    )
    assert optimizer.get_parameter_groups() == []


def test_name():
    for name in ImplementedOptimizer:
        config = globals()[f"{name.value}Config"]
        c = config()
        assert c.name_ == name.value
