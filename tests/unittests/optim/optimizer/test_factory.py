from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from torch.optim import Adagrad

from clinicadl.optim.optimizer import (
    ImplementedOptimizer,
    create_optimizer_config,
    get_optimizer,
)
from clinicadl.optim.optimizer.factory import (
    _get_params_in_group,
    _get_params_not_in_group,
    _regroup_args,
)


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


def test_get_optimizer(network):
    for optimizer in ImplementedOptimizer:
        config = create_optimizer_config(optimizer=optimizer)()
        optimizer, _ = get_optimizer(network, config)
        assert len(optimizer.param_groups) == 1

    config = create_optimizer_config("Adagrad")(
        lr=1e-5,
        weight_decay={"final.dense3.weight": 1.0, "dense1": 0.1},
        lr_decay={"dense1": 10, "ELSE": 100},
        eps={"ELSE": 1.0},
    )
    optimizer, updated_config = get_optimizer(network, config)
    assert isinstance(optimizer, Adagrad)
    assert len(optimizer.param_groups) == 3

    assert len(optimizer.param_groups[0]["params"]) == 2
    assert len(optimizer.param_groups[1]["params"]) == 1
    assert len(optimizer.param_groups[2]["params"]) == 5

    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-5
    assert optimizer.param_groups[2]["lr"] == 1e-5

    assert optimizer.param_groups[0]["lr_decay"] == 10
    assert optimizer.param_groups[1]["lr_decay"] == 100
    assert optimizer.param_groups[2]["lr_decay"] == 100

    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 1.0
    assert optimizer.param_groups[2]["weight_decay"] == 0.0

    assert optimizer.param_groups[0]["eps"] == 1.0
    assert optimizer.param_groups[1]["eps"] == 1.0
    assert optimizer.param_groups[2]["eps"] == 1.0

    assert not optimizer.param_groups[0]["differentiable"]
    assert not optimizer.param_groups[1]["differentiable"]
    assert not optimizer.param_groups[2]["differentiable"]

    assert updated_config.optimizer == "Adagrad"
    assert updated_config.lr == 1e-5
    assert updated_config.lr_decay == {"dense1": 10, "ELSE": 100}
    assert updated_config.weight_decay == {"dense1": 0.1, "final.dense3.weight": 1.0}
    assert updated_config.initial_accumulator_value == 0
    assert updated_config.eps == {"ELSE": 1.0}
    assert updated_config.foreach is None
    assert not updated_config.maximize
    assert not updated_config.differentiable

    # special cases 1
    config = create_optimizer_config("Adagrad")(
        lr_decay={"ELSE": 100},
    )
    optimizer, _ = get_optimizer(network, config)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr_decay"] == 100

    # special cases 2
    config = create_optimizer_config("Adagrad")(
        lr_decay={"conv1": 100, "dense1": 10, "final": 1},
    )
    optimizer, _ = get_optimizer(network, config)
    assert len(optimizer.param_groups) == 3


def test_regroup_args():
    args = {
        "weight_decay": {"params_0": 0.0, "params_1": 1.0},
        "alpha": {"params_1": 0.5, "ELSE": 0.1},
        "momentum": {"params_3": 3.0},
        "betas": (0.1, 0.1),
    }
    args_groups, args_global = _regroup_args(args)
    assert args_groups == {
        "params_0": {"weight_decay": 0.0},
        "params_1": {"alpha": 0.5, "weight_decay": 1.0},
        "params_3": {"momentum": 3.0},
    }
    assert args_global == {"betas": (0.1, 0.1), "alpha": 0.1}

    args_groups, args_global = _regroup_args({"betas": (0.1, 0.1)})
    assert len(args_groups) == 0

    args_groups, args_global = _regroup_args(
        {"weight_decay": {"params_0": 0.0, "params_1": 1.0}}
    )
    assert len(args_global) == 0


def test_get_params_in_block(network):
    generator, list_layers = _get_params_in_group(network, "dense1")
    assert next(iter(generator)).shape == torch.Size((10, 10))
    assert next(iter(generator)).shape == torch.Size((10,))
    assert sorted(list_layers) == sorted(["dense1.weight", "dense1.bias"])

    generator, list_layers = _get_params_in_group(network, "dense1.weight")
    assert next(iter(generator)).shape == torch.Size((10, 10))
    assert sum(1 for _ in generator) == 0
    assert sorted(list_layers) == sorted(["dense1.weight"])

    generator, list_layers = _get_params_in_group(network, "final.dense3")
    assert next(iter(generator)).shape == torch.Size((3, 5))
    assert next(iter(generator)).shape == torch.Size((3,))
    assert sorted(list_layers) == sorted(["final.dense3.weight", "final.dense3.bias"])

    generator, list_layers = _get_params_in_group(network, "final")
    assert sum(1 for _ in generator) == 4
    assert sorted(list_layers) == sorted(
        [
            "final.dense2.weight",
            "final.dense2.bias",
            "final.dense3.weight",
            "final.dense3.bias",
        ]
    )


def test_find_params_not_in_group(network):
    params = _get_params_not_in_group(
        network,
        [
            "final.dense2.weight",
            "final.dense2.bias",
            "conv1.bias",
            "final.dense3.weight",
            "dense1.weight",
            "dense1.bias",
        ],
    )
    assert next(iter(params)).shape == torch.Size((1, 1, 3, 3))
    assert next(iter(params)).shape == torch.Size((3,))
    assert sum(1 for _ in params) == 0  # no more params
