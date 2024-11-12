import pytest
from pydantic import ValidationError

from clinicadl.losses.config import (
    BCELossConfig,
    BCEWithLogitsLossConfig,
    CrossEntropyLossConfig,
    HuberLossConfig,
    KLDivLossConfig,
    L1LossConfig,
    MSELossConfig,
    MultiMarginLossConfig,
    NLLLossConfig,
    SmoothL1LossConfig,
    create_loss_function_config,
)
from clinicadl.losses.enum import ImplementedLoss

BAD_INPUTS = [
    ("reduction", "none"),
    ("weight", [1, -1, 2]),
    ("ignore_index", -1),
    ("label_smoothing", 1.1),
    ("pos_weight", [1, -1, 2]),
    ("delta", 0.0),
    ("beta", -0.1),
    ("p", 3),
    ("margin", None),
    ("log_target", None),
]

GOOD_INPUTS = [
    ("reduction", "mean"),
    ("weight", [1, 1, 2]),
    ("ignore_index", -100),
    ("label_smoothing", 0.5),
    ("pos_weight", [1, 1, 2]),
    ("delta", 0.1),
    ("beta", 0),
    ("p", 1),
    ("margin", -0.5),
    ("log_target", True),
    ("reduction", "sum"),
    ("ignore_index", 0),
    ("p", 2),
    ("log_target", False),
]


@pytest.mark.parametrize(
    "arg,value",
    BAD_INPUTS,
)
def test_validation_fail(arg, value):
    for loss in ImplementedLoss:
        config = create_loss_function_config(loss)
        fields = config.model_fields
        if arg in fields:
            with pytest.raises(ValidationError):
                config(**{arg: value})


@pytest.mark.parametrize(
    "arg,value",
    GOOD_INPUTS,
)
def test_validation_pass(arg, value):
    for loss in ImplementedLoss:
        config = create_loss_function_config(loss)
        fields = config.model_fields

        if arg in fields:
            if (loss == "BCELoss" or loss == "BCEWithLogitsLoss") and arg == "weight":
                value_ = None
            else:
                value_ = value

            c = config(**{arg: value_})
            assert getattr(c, arg) == value_


def test_weight_validator():
    with pytest.raises(ValidationError):
        BCELossConfig(weight=[1, 2])


@pytest.mark.parametrize(
    "name,config",
    [
        ("BCELoss", BCELossConfig),
        ("BCEWithLogitsLoss", BCEWithLogitsLossConfig),
        ("CrossEntropyLoss", CrossEntropyLossConfig),
        ("HuberLoss", HuberLossConfig),
        ("KLDivLoss", KLDivLossConfig),
        ("L1Loss", L1LossConfig),
        ("MSELoss", MSELossConfig),
        ("MultiMarginLoss", MultiMarginLossConfig),
        ("NLLLoss", NLLLossConfig),
        ("SmoothL1Loss", SmoothL1LossConfig),
    ],
)
def test_create_loss_function_config(name, config):
    assert create_loss_function_config(name) == config
