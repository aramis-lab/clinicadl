import pytest

from clinicadl.optim.lr_schedulers.config import *


@pytest.mark.parametrize(
    "args,name,config,type_",
    [
        ({}, "ConstantLR", ConstantLRConfig, "epoch-based"),
        ({"gamma": 1}, "ExponentialLR", ExponentialLRConfig, "epoch-based"),
        ({}, "LinearLR", LinearLRConfig, "epoch-based"),
        ({"step_size": 1}, "StepLR", StepLRConfig, "epoch-based"),
        ({"milestones": [1, 2]}, "MultiStepLR", MultiStepLRConfig, "epoch-based"),
        ({}, "PolynomialLR", PolynomialLRConfig, "epoch-based"),
        ({}, "ReduceLROnPlateau", ReduceLROnPlateauConfig, "loss-based"),
        (
            {"max_lr": 1, "total_steps": 10},
            "OneCycleLR",
            OneCycleLRConfig,
            "step-based",
        ),
    ],
)
def test_get_lr_scheduler_from_config(args, name, config, type_):
    c = get_lr_scheduler_config(name, **args)

    assert c.name == name
    assert isinstance(c, config)
    assert c.scheduler_type() == type_

    if name == "StepLR":
        config = get_lr_scheduler_config(
            "StepLR",
            step_size=1,
        )
        assert config.name == "StepLR"
        assert config.step_size == 1
        assert config.gamma == 0.1

        with pytest.raises(ValueError):
            get_lr_scheduler_config("abc", step_size=1)
