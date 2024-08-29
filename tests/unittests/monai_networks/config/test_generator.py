import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.generator import GeneratorConfig


@pytest.fixture
def dummy_arguments():
    args = {"latent_shape": (5,), "channels": (2, 4)}
    return args


@pytest.fixture(
    params=[
        {"start_shape": (3,), "strides": (1, 1)},
        {"start_shape": (1, 3), "strides": (1, 1), "dropout": 1.1},
        {"start_shape": (1, 3), "strides": (1, 1), "kernel_size": 4},
        {"start_shape": (1, 3), "strides": (1, 1), "kernel_size": (3, 3)},
        {"start_shape": (1, 3), "strides": (1, 2, 3)},
        {"start_shape": (1, 3), "strides": (1, (1, 2))},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        GeneratorConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {"start_shape": (1, 3), "strides": (1, 1), "dropout": 0.5, "kernel_size": 5},
        {
            "start_shape": (1, 3, 3, 3),
            "strides": ((1, 2, 3), 1),
            "kernel_size": (3, 3, 3),
        },
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    GeneratorConfig(**good_inputs)


def test_GeneratorConfig():
    config = GeneratorConfig(
        latent_shape=(3,),
        start_shape=(1, 3),
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3,),
        num_res_units=1,
        act="SIGMOID",
        dropout=0.1,
        bias=False,
    )
    assert config.network == "Generator"
    assert config.latent_shape == (3,)
    assert config.start_shape == (1, 3)
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3,)
    assert config.num_res_units == 1
    assert config.act == "sigmoid"
    assert config.norm == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias
