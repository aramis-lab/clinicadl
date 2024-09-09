from collections import OrderedDict

import pytest
import torch.nn as nn


@pytest.fixture
def network():
    network = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 1, kernel_size=3)),
                ("dense1", nn.Linear(10, 10)),
            ]
        )
    )
    network.add_module(
        "final",
        nn.Sequential(
            OrderedDict([("dense2", nn.Linear(10, 5)), ("dense3", nn.Linear(5, 3))])
        ),
    )
    return network


def test_get_optimizer(network):
    from torch.optim import Adagrad

    from clinicadl.optim.optimizer import (
        ImplementedOptimizer,
        OptimizerConfig,
        get_optimizer,
    )

    for optimizer in [e.value for e in ImplementedOptimizer]:
        config = OptimizerConfig(optimizer=optimizer)
        optimizer, _ = get_optimizer(network, config)
        assert len(optimizer.param_groups) == 1

    config = OptimizerConfig(
        optimizer="Adagrad",
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

    # special case : only ELSE
    config = OptimizerConfig(
        optimizer="Adagrad",
        lr_decay={"ELSE": 100},
    )
    optimizer, _ = get_optimizer(network, config)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr_decay"] == 100

    # special case : the params mentioned form all the network
    config = OptimizerConfig(
        optimizer="Adagrad",
        lr_decay={"conv1": 100, "dense1": 10, "final": 1},
    )
    optimizer, _ = get_optimizer(network, config)
    assert len(optimizer.param_groups) == 3

    # special case : no ELSE mentioned
    config = OptimizerConfig(
        optimizer="Adagrad",
        lr_decay={"conv1": 100},
    )
    optimizer, _ = get_optimizer(network, config)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr_decay"] == 100
    assert optimizer.param_groups[1]["lr_decay"] == 0


def test_regroup_args():
    from clinicadl.optim.optimizer.factory import _regroup_args

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
