from copy import deepcopy

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
from clinicadl.optim.lr_scheduler.enum import ImplementedLRScheduler

MANDATORY_FIELDS = {"step_size": 1, "milestones": [1, 2]}
BAD_INPUTS = [
    ("milestones", [4, 2, 4]),
    ("gamma", 0),
    ("last_epoch", -2),
    ("step_size", 0),
    ("factor", 0),
    ("total_iters", 0),
    ("start_factor", 0),
    ("end_factor", 0),
    ("mode", "abc"),
    ("patience", -1),
    ("threshold", -1),
    ("threshold_mode", "abc"),
    ("cooldown", -1),
    ("eps", -0.1),
    ("min_lr", -0.1),
    ("milestones", [0, 1]),
]

GOOD_INPUTS = [
    ("gamma", 0.1),
    ("last_epoch", -1),
    ("factor", 0.1),
    ("total_iters", 1),
    ("start_factor", 0.1),
    ("end_factor", 0.2),
    ("mode", "min"),
    ("patience", 0),
    ("threshold", 0),
    ("threshold_mode", "abs"),
    ("cooldown", 0),
    ("eps", 0),
    ("min_lr", 0),
]


@pytest.mark.parametrize("arg,value", BAD_INPUTS)
def test_validation_fail(arg, value):
    for scheduler in ImplementedLRScheduler:
        config = create_lr_scheduler_config(scheduler)
        fields = config.model_fields
        if arg in fields:
            mandatory_inputs = deepcopy(MANDATORY_FIELDS)
            if arg in mandatory_inputs:
                del mandatory_inputs[arg]

            with pytest.raises(ValidationError):
                config(**{arg: value}, **mandatory_inputs)

            # test dict inputs for min_lr
            if arg == "min_lr":
                value_dict = {"group_1": value, "ELSE": value}
                with pytest.raises(ValidationError):
                    config(**{arg: value_dict})


@pytest.mark.parametrize(
    "arg,value",
    GOOD_INPUTS,
)
def test_validation_pass(arg, value):
    for scheduler in ImplementedLRScheduler:
        config = create_lr_scheduler_config(scheduler)
        fields = config.model_fields

        if arg in fields:
            mandatory_inputs = deepcopy(MANDATORY_FIELDS)
            if arg in mandatory_inputs:
                del mandatory_inputs[arg]

            c = config(**{arg: value}, **mandatory_inputs)
            assert getattr(c, arg) == value

            # test dict inputs
            if arg == "min_lr":
                value_dict = {"group_1": value, "ELSE": value}
                c = config(**{arg: value_dict}, **mandatory_inputs)
                assert getattr(c, "min_lr") == value_dict


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


def test_minr_lr_validator():
    with pytest.raises(ValidationError):
        ReduceLROnPlateauConfig(min_lr={"params1": 0.1})
    ReduceLROnPlateauConfig(min_lr={"params1": 0.1, "ELSE": 0.2})


def test_get_all_groups():
    config = ReduceLROnPlateauConfig(
        min_lr={"params1": 0.1, "params3": 0.7, "ELSE": 0.2},
    )
    assert config.get_all_groups() == {"params1", "params3", "ELSE"}

    config.min_lr = 0.1
    assert config.get_all_groups() == set()

    config = ConstantLRConfig()
    assert config.get_all_groups() == set()
