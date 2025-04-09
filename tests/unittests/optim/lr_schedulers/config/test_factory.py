import pytest

from clinicadl.optim.lr_schedulers.config import *


@pytest.mark.parametrize(
    "args,name,config",
    [
        ({}, "ConstantLR", ConstantLRConfig),
        ({"gamma": 1}, "ExponentialLR", ExponentialLRConfig),
        ({}, "LinearLR", LinearLRConfig),
        ({"step_size": 1}, "StepLR", StepLRConfig),
        ({"milestones": [1, 2]}, "MultiStepLR", MultiStepLRConfig),
        ({}, "PolynomialLR", PolynomialLRConfig),
        ({}, "ReduceLROnPlateau", ReduceLROnPlateauConfig),
        ({"max_lr": 1, "total_steps": 10}, "OneCycleLR", OneCycleLRConfig),
    ],
)
def test_get_lr_scheduler_from_config(args, name, config):
    c = get_lr_scheduler_config(name, **args)

    assert c.name == name
    assert isinstance(c, config)

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
