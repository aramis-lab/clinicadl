import pytest
from pydantic import ValidationError

from clinicadl.optimization.optimizer.config import (
    AdadeltaConfig,
    AdagradConfig,
    AdamConfig,
    RMSpropConfig,
    SGDConfig,
    create_optimizer_config,
)

BAD_INPUTS = {
    "lr": 0,
    "rho": 1.1,
    "eps": -0.1,
    "weight_decay": -0.1,
    "lr_decay": -0.1,
    "initial_accumulator_value": -0.1,
    "betas": (0.9, 1.0),
    "alpha": 1.1,
    "momentum": -0.1,
    "dampening": 0.1,
}

GOOD_INPUTS_1 = {
    "lr": 0.1,
    "rho": 0,
    "eps": 0,
    "weight_decay": 0,
    "foreach": None,
    "capturable": False,
    "maximize": True,
    "differentiable": False,
    "fused": None,
    "lr_decay": 0,
    "initial_accumulator_value": 0,
    "betas": (0.0, 0.0),
    "amsgrad": True,
    "alpha": 0.0,
    "momentum": 0,
    "centered": True,
    "dampening": 0,
    "nesterov": True,
    "freeze": "params1",
}

GOOD_INPUTS_2 = {
    "foreach": True,
    "fused": False,
    "freeze": ["params1", "params2"],
}


@pytest.mark.parametrize(
    "config",
    [
        AdadeltaConfig,
        AdagradConfig,
        AdamConfig,
        RMSpropConfig,
        SGDConfig,
    ],
)
def test_validation_fail(config):
    fields = config.model_fields
    inputs = {key: value for key, value in BAD_INPUTS.items() if key in fields}
    with pytest.raises(ValidationError):
        config(**inputs)

    # test dict inputs
    inputs = {
        key: {"group_1": value} for key, value in inputs.items() if key != "freeze"
    }
    with pytest.raises(ValidationError):
        config(**inputs)


@pytest.mark.parametrize(
    "config,name",
    [
        (AdadeltaConfig, "Adadelta"),
        (AdagradConfig, "Adagrad"),
        (AdamConfig, "Adam"),
        (RMSpropConfig, "RMSprop"),
        (SGDConfig, "SGD"),
    ],
)
@pytest.mark.parametrize(
    "good_inputs",
    [
        GOOD_INPUTS_1,
        GOOD_INPUTS_2,
    ],
)
def test_validation_pass(config, name, good_inputs):
    fields = config.model_fields
    inputs = {key: value for key, value in good_inputs.items() if key in fields}
    c = config(**inputs)
    for arg, value in inputs.items():
        if arg == "freeze":
            assert getattr(c, arg) == value if isinstance(value, list) else [value]
        else:
            assert getattr(c, arg) == value
    assert c.name == name

    # test dict inputs
    inputs = {
        key: {"group_1": value, "ELSE": value}
        for key, value in inputs.items()
        if key != "freeze"
    }
    c = config(**inputs)
    for arg, value in inputs.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("Adadelta", AdadeltaConfig),
        ("Adagrad", AdagradConfig),
        ("Adam", AdamConfig),
        ("RMSprop", RMSpropConfig),
        ("SGD", SGDConfig),
    ],
)
def test_create_optimizer_config(name, expected_class):
    config = create_optimizer_config(name)
    assert config == expected_class


def test_get_all_groups():
    c = SGDConfig(
        lr={"params1": 0.1, "params3": 0.7, "ELSE": 0.2},
        weight_decay={"params2": 0.3, "ELSE": 0.5},
    )
    assert c.get_all_groups() == {"params1", "params2", "params3", "ELSE"}


def test_get_check_else():
    with pytest.raises(ValidationError):
        SGDConfig(lr={"params1": 0.1, "ELSE": 0.2}, nesterov={"params2": False})
    SGDConfig(
        lr={"params1": 0.1, "ELSE": 0.2}, nesterov={"params2": False, "ELSE": True}
    )


def test_check_param_groups():
    with pytest.raises(ValidationError):
        SGDConfig(
            lr={"params1": 0.1, "ELSE": 0.2},
            nesterov={"params2": False, "ELSE": True},
            freeze="params2",
        )
    SGDConfig(
        lr={"params1": 0.1, "ELSE": 0.2},
        nesterov={"params2": False, "ELSE": True},
        freeze="params3",
    )
