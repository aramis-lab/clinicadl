from collections import OrderedDict

import pytest
import torch.nn as nn
from torch.optim import Adagrad

from clinicadl.optim.optimizers.config import (
    ImplementedOptimizer,
    create_optimizer_config,
)
from clinicadl.optim.optimizers.factory import (
    _regroup_args_by_param_group,
    get_optimizer_config,
    get_optimizer_from_config,
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


def test_get_optimizer_from_config(network):
    # test all optimizers
    for optimizer in ImplementedOptimizer:
        config = create_optimizer_config(optimizer=optimizer)()
        optimizer, _ = get_optimizer_from_config(config=config, network=network)
        assert len(optimizer.param_groups) == 1

    # test arguments
    config = create_optimizer_config(optimizer="Adagrad")(
        lr=1e-5,
        weight_decay={"final.dense3.weight": 1.0, "dense1": 0.1, "ELSE": 0.2},
        lr_decay={"final.dense3.bias": 1, "dense1": 10, "ELSE": 100},
        eps={"ELSE": 1.0},
    )
    optimizer, updated_config = get_optimizer_from_config(config, network)
    assert isinstance(optimizer, Adagrad)
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

    # check that config is not modified
    assert updated_config.name == "Adagrad"
    assert updated_config.lr == 1e-5
    assert updated_config.lr_decay == {
        "final.dense3.bias": 1,
        "dense1": 10,
        "ELSE": 100,
    }
    assert updated_config.weight_decay == {
        "final.dense3.weight": 1.0,
        "dense1": 0.1,
        "ELSE": 0.2,
    }
    assert updated_config.initial_accumulator_value == 0
    assert updated_config.eps == {"ELSE": 1.0}
    assert updated_config.foreach is None
    assert not updated_config.maximize
    assert not updated_config.differentiable

    # special case
    config = create_optimizer_config("Adagrad")(
        lr_decay={"ELSE": 100},
    )
    optimizer, _ = get_optimizer_from_config(config, network=network)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr_decay"] == 100

    # test freeze
    config = create_optimizer_config("Adagrad")(
        lr_decay={"final": 10, "ELSE": 100}, freeze="conv1"
    )
    optimizer, _ = get_optimizer_from_config(config, network=network)
    assert len(optimizer.param_groups) == 2
    for param in network.conv1.parameters():
        assert not param.requires_grad
    for param in network.final.parameters():
        assert param.requires_grad
    for param in network.dense1.parameters():
        assert param.requires_grad


def test_get_optimizer_config():
    config = get_optimizer_config(
        "Adagrad", lr=0.1, weight_decay={"param1": 0.1, "ELSE": 0}
    )
    assert config.name == "Adagrad"
    assert config.lr == 0.1
    assert config.weight_decay == {"param1": 0.1, "ELSE": 0}
    assert config.lr_decay == 0

    with pytest.raises(ValueError):
        get_optimizer_config("abc", lr=0.1, weight_decay={"param1": 0.1, "ELSE": 0})


def test_regroup_args_by_param_group():
    args = {
        "weight_decay": {"params_0": 0.0, "params_1": 1.0},
        "alpha": {"params_1": 0.5, "ELSE": 0.1},
        "momentum": {"params_3": 3.0},
        "betas": (0.1, 0.1),
    }
    args_groups, args_global = _regroup_args_by_param_group(args)
    assert args_groups == {
        "params_0": {"weight_decay": 0.0},
        "params_1": {"alpha": 0.5, "weight_decay": 1.0},
        "params_3": {"momentum": 3.0},
    }
    assert args_global == {"betas": (0.1, 0.1), "alpha": 0.1}

    args_groups, args_global = _regroup_args_by_param_group({"betas": (0.1, 0.1)})
    assert len(args_groups) == 0

    args_groups, args_global = _regroup_args_by_param_group(
        {"weight_decay": {"params_0": 0.0, "params_1": 1.0}}
    )
    assert len(args_global) == 0
