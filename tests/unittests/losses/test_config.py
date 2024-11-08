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

BAD_INPUTS = {
    "reduction": "none",
    "weight": [1, -1, 2],
    "ignore_index": -1,
    "label_smoothing": 1.1,
    "pos_weight": [1, -1, 2],
    "delta": 0.0,
    "beta": -0.1,
    "p": 3,
    "margin": None,
    "log_target": None,
}

GOOD_INPUTS_1 = {
    "reduction": "mean",
    "weight": [1, 1, 2],
    "ignore_index": -100,
    "label_smoothing": 0.5,
    "pos_weight": [1, 1, 2],
    "delta": 0.1,
    "beta": 0,
    "p": 1,
    "margin": -0.5,
    "log_target": True,
}

GOOD_INPUTS_2 = {
    "reduction": "sum",
    "ignore_index": 0,
    "p": 2,
    "log_target": False,
}


def test_validation_fail():
    for loss in ImplementedLoss:
        config = create_loss_function_config(loss)
        fields = config.model_fields
        inputs = {key: value for key, value in BAD_INPUTS.items() if key in fields}
        for input, value in inputs.items():
            with pytest.raises(ValidationError):
                config(**{input: value})


@pytest.mark.parametrize(
    "good_inputs",
    [
        GOOD_INPUTS_1,
        GOOD_INPUTS_2,
    ],
)
def test_validation_pass(good_inputs):
    for loss in ImplementedLoss:
        config = create_loss_function_config(loss)
        fields = config.model_fields
        inputs = {key: value for key, value in good_inputs.items() if key in fields}

        if (
            loss == "BCELoss" or loss == "BCEWithLogitsLoss"
        ) and "weight" in good_inputs:
            with pytest.raises(ValidationError):
                config(**{"weight": good_inputs["weight"]})
            inputs["weight"] = None

        c = config(**inputs)
        for arg, value in inputs.items():
            assert getattr(c, arg) == value
        assert c.name == loss.value


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
