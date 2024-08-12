from collections import OrderedDict

import pytest
import torch.nn as nn
from pydantic import ValidationError

from clinicadl.optim import OptimizationConfig
from clinicadl.optim.lr_scheduler import LRSchedulerConfig
from clinicadl.optim.optimizer import OptimizerConfig


def test_OptimizationConfig():
    config = OptimizationConfig(
        accumulation_steps=2,
        lr_scheduler=LRSchedulerConfig(scheduler="StepLR", step_size=1),
    )
    config.epochs = 5
    config.early_stopping.lower_bound = 0
    config.optimizer.optimizer = "SGD"

    assert config.accumulation_steps == 2
    assert config.lr_scheduler.scheduler == "StepLR"
    assert config.lr_scheduler.step_size == 1
    assert config.epochs == 5
    assert config.early_stopping.lower_bound == 0
    assert config.optimizer.optimizer == "SGD"


def test_model_validation():
    optimizer_config = OptimizerConfig(lr={"param_0": 1}, weight_decay={"param_1": 1})

    scheduler_config = LRSchedulerConfig(min_lr={"param_0": 1, "param_2": 10})
    with pytest.raises(ValidationError):
        OptimizationConfig(optimizer=optimizer_config, lr_scheduler=scheduler_config)

    scheduler_config = LRSchedulerConfig(min_lr={"param_0": 1})
    with pytest.raises(ValidationError):
        OptimizationConfig(optimizer=optimizer_config, lr_scheduler=scheduler_config)

    scheduler_config = LRSchedulerConfig(min_lr={"param_0": 1, "param_1": 10})
    OptimizationConfig(optimizer=optimizer_config, lr_scheduler=scheduler_config)


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


def test_group_order(network):
    from clinicadl.optim.lr_scheduler import get_lr_scheduler
    from clinicadl.optim.optimizer import get_optimizer

    optimizer_config = OptimizerConfig(
        lr={"final.dense2.weight": 0.1, "conv1": 1},
        weight_decay={"ELSE": 10, "final.dense3": 1},
    )
    scheduler_config = LRSchedulerConfig(
        scheduler="ReduceLROnPlateau",
        min_lr={
            "ELSE": 0.1,
            "conv1": 1,
            "final.dense2.weight": 10,
            "final.dense3": 100,
        },
    )
    config = OptimizationConfig(
        optimizer=optimizer_config, lr_scheduler=scheduler_config
    )

    optimizer, _ = get_optimizer(network, config.optimizer)
    scheduler, _ = get_lr_scheduler(optimizer, config.lr_scheduler)

    assert scheduler.min_lrs == [1, 10, 100, 0.1]

    assert len(optimizer.param_groups) == 4
    assert len(optimizer.param_groups[0]["params"]) == 2
    assert len(optimizer.param_groups[1]["params"]) == 1
    assert len(optimizer.param_groups[2]["params"]) == 2
    assert len(optimizer.param_groups[3]["params"]) == 3

    assert optimizer.param_groups[0]["lr"] == 1
    assert optimizer.param_groups[1]["lr"] == 0.1
    assert optimizer.param_groups[2]["lr"] == 0.001
    assert optimizer.param_groups[3]["lr"] == 0.001

    assert optimizer.param_groups[0]["weight_decay"] == 10
    assert optimizer.param_groups[1]["weight_decay"] == 10
    assert optimizer.param_groups[2]["weight_decay"] == 1
    assert optimizer.param_groups[3]["weight_decay"] == 10
