import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.classifier import (
    ClassifierConfig,
    CriticConfig,
    DiscriminatorConfig,
)


@pytest.fixture
def dummy_arguments():
    args = {
        "classes": 2,
        "channels": [2, 4],
    }
    return args


@pytest.fixture(
    params=[
        {"in_shape": (3,), "strides": (1, 1)},
        {"in_shape": (1, 3, 3), "strides": (1, 1), "dropout": 1.1},
        {"in_shape": (1, 3, 3), "strides": (1, 1), "kernel_size": 4},
        {"in_shape": (1, 3, 3), "strides": (1, 1), "kernel_size": (3,)},
        {"in_shape": (1, 3, 3), "strides": (1, 1), "kernel_size": (3, 3, 3)},
        {"in_shape": (1, 3, 3), "strides": (1, 2, 3)},
        {"in_shape": (1, 3, 3), "strides": (1, (1, 2, 3))},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        ClassifierConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        CriticConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        DiscriminatorConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {"in_shape": (1, 3, 3), "strides": (1, 1), "dropout": 0.5, "kernel_size": 5},
        {"in_shape": (1, 3, 3), "strides": ((1, 2), 1), "kernel_size": (3, 3)},
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    ClassifierConfig(**good_inputs)
    CriticConfig(**good_inputs)
    DiscriminatorConfig(**good_inputs)


def test_ClassifierConfig():
    config = ClassifierConfig(
        in_shape=(1, 3, 3),
        classes=2,
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3, 5),
        num_res_units=1,
        act=("ELU", {"alpha": 2.0}),
        dropout=0.1,
        bias=False,
        last_act=None,
    )
    assert config.in_shape == (1, 3, 3)
    assert config.classes == 2
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3, 5)
    assert config.num_res_units == 1
    assert config.act == ("elu", {"alpha": 2.0})
    assert config.norm == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias
    assert config.last_act is None


def test_CriticConfig():
    config = CriticConfig(
        in_shape=(1, 3, 3),
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3, 5),
        num_res_units=1,
        act=("ELU", {"alpha": 2.0}),
        dropout=0.1,
        bias=False,
    )
    assert config.in_shape == (1, 3, 3)
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3, 5)
    assert config.num_res_units == 1
    assert config.act == ("elu", {"alpha": 2.0})
    assert config.norm == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias


def test_DiscriminatorConfig():
    config = DiscriminatorConfig(
        in_shape=(1, 3, 3),
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3, 5),
        num_res_units=1,
        act=("ELU", {"alpha": 2.0}),
        dropout=0.1,
        bias=False,
        last_act=("eLu", {"alpha": 0.5}),
    )
    assert config.in_shape == (1, 3, 3)
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3, 5)
    assert config.num_res_units == 1
    assert config.act == ("elu", {"alpha": 2.0})
    assert config.norm == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias
    assert config.last_act == ("elu", {"alpha": 0.5})
