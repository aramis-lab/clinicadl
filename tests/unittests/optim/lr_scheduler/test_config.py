import pytest
from pydantic import ValidationError

from clinicadl.optim.lr_scheduler.config import (
    ConstantLRConfig,
    LinearLRConfig,
    MultiStepLRConfig,
    ReduceLROnPlateauConfig,
    StepLRConfig,
    create_lr_scheduler_config,
)

BAD_INPUTS = {
    "milestones": [3, 2, 4],
    "gamma": 0,
    "last_epoch": -2,
    "step_size": 0,
    "factor": 0,
    "total_iters": 0,
    "start_factor": 0,
    "end_factor": 0,
    "mode": "abc",
    "patience": 0,
    "threshold": -1,
    "threshold_mode": "abc",
    "cooldown": -1,
    "eps": -0.1,
    "min_lr": -0.1,
}

GOOD_INPUTS = {
    "milestones": [1, 4, 5],
    "gamma": 0.1,
    "last_epoch": -1,
    "step_size": 1,
    "factor": 0.1,
    "total_iters": 1,
    "start_factor": 0.1,
    "end_factor": 0.2,
    "mode": "min",
    "patience": 1,
    "threshold": 0,
    "threshold_mode": "abs",
    "cooldown": 0,
    "eps": 0,
    "min_lr": 0,
}


@pytest.mark.parametrize(
    "config",
    [
        ConstantLRConfig,
        LinearLRConfig,
        MultiStepLRConfig,
        ReduceLROnPlateauConfig,
        StepLRConfig,
    ],
)
def test_validation_fail(config):
    fields = config.model_fields
    inputs = {key: value for key, value in BAD_INPUTS.items() if key in fields}
    with pytest.raises(ValidationError):
        config(**inputs)

    # test dict inputs for min_lr
    if "min_lr" in inputs:
        inputs["min_lr"] = {"group_1": inputs["min_lr"]}
        with pytest.raises(ValidationError):
            config(**inputs)


def test_validation_fail_special():
    with pytest.raises(ValidationError):
        MultiStepLRConfig(milestones=[0, 1])


@pytest.mark.parametrize(
    "config",
    [
        ConstantLRConfig,
        LinearLRConfig,
        MultiStepLRConfig,
        ReduceLROnPlateauConfig,
        StepLRConfig,
    ],
)
def test_validation_pass(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS.items() if key in fields}
    c = config(**inputs)
    for arg, value in inputs.items():
        assert getattr(c, arg) == value

    # test dict inputs
    if "min_lr" in inputs:
        inputs["min_lr"] = {"group_1": inputs["min_lr"]}
        c = config(**inputs)
        assert getattr(c, "min_lr") == inputs["min_lr"]


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("ConstantLR", ConstantLRConfig),
        ("LinearLR", LinearLRConfig),
        ("MultiStepLR", MultiStepLRConfig),
        ("ReduceLROnPlateau", ReduceLROnPlateauConfig),
        ("StepLR", StepLRConfig),
    ],
)
def test_create_optimizer_config(name, expected_class):
    config = create_lr_scheduler_config(name)
    assert config == expected_class
