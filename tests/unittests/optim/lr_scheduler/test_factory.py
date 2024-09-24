from collections import OrderedDict

import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from clinicadl.optim.lr_scheduler import (
    ImplementedLRScheduler,
    create_lr_scheduler_config,
    get_lr_scheduler,
)


def test_get_lr_scheduler():
    network = nn.Sequential(
        OrderedDict(
            [
                ("linear1", nn.Linear(4, 3)),
                ("linear2", nn.Linear(3, 2)),
                ("linear3", nn.Linear(2, 1)),
            ]
        )
    )
    optimizer = SGD(
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

    args = {"step_size": 1, "milestones": [1, 2]}
    for scheduler in ImplementedLRScheduler:
        config = create_lr_scheduler_config(scheduler=scheduler)(**args)
        _ = get_lr_scheduler(optimizer, config)

    config = create_lr_scheduler_config(scheduler="ReduceLROnPlateau")(
        mode="max",
        factor=0.123,
        threshold=1e-1,
        cooldown=3,
        min_lr={"linear2": 0.01, "linear1": 0.1},
    )
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.123
    assert scheduler.patience == 10
    assert scheduler.threshold == 1e-1
    assert scheduler.threshold_mode == "rel"
    assert scheduler.cooldown == 3
    assert scheduler.min_lrs == [0.1, 0.01, 0.0]
    assert scheduler.eps == 1e-8

    assert updated_config.scheduler == "ReduceLROnPlateau"
    assert updated_config.mode == "max"
    assert updated_config.factor == 0.123
    assert updated_config.patience == 10
    assert updated_config.threshold == 1e-1
    assert updated_config.threshold_mode == "rel"
    assert updated_config.cooldown == 3
    assert updated_config.min_lr == {"linear2": 0.01, "linear1": 0.1}
    assert updated_config.eps == 1e-8

    config.min_lr = {"ELSE": 1, "linear2": 0.01, "linear1": 0.1}
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert scheduler.min_lrs == [0.1, 0.01, 1]

    config.min_lr = 1
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert scheduler.min_lrs == [1.0, 1.0, 1.0]

    # no lr scheduler
    config = create_lr_scheduler_config(None)()
    scheduler, updated_config = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, LambdaLR)
    assert updated_config.scheduler is None
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [1.0, 10.0, 10.0]
