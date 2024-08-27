import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.autoencoder import (
    AutoEncoderConfig,
    VarAutoEncoderConfig,
)


@pytest.fixture
def dummy_arguments():
    args = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [2, 4],
        "latent_size": 16,
    }
    return args


@pytest.fixture(
    params=[
        {"in_shape": (1, 10, 10), "strides": (1, 1), "dropout": 1.1},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "kernel_size": 4},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "kernel_size": (3,)},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "kernel_size": (3, 3, 3)},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "up_kernel_size": 4},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "up_kernel_size": (3,)},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "up_kernel_size": (3, 3, 3)},
        {
            "in_shape": (1, 10, 10),
            "strides": (1, 1),
            "inter_channels": (2, 2),
            "inter_dilations": (2,),
        },
        {"in_shape": (1, 10, 10), "strides": (1, 1), "inter_dilations": (2, 2)},
        {"in_shape": (1, 10, 10), "strides": (1, 1), "padding": (1, 1, 1)},
        {"in_shape": (1, 10, 10), "strides": (1, 2, 3)},
        {"in_shape": (1, 10, 10), "strides": (1, (1, 2, 3))},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture(
    params=[
        {"in_shape": (1,), "strides": (1, 1)},
        {"in_shape": (1, 10), "strides": (1, 1)},
    ]
)
def bad_inputs_vae(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        AutoEncoderConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        VarAutoEncoderConfig(**bad_inputs)


def test_fails_validations_vae(bad_inputs_vae):
    with pytest.raises(ValidationError):
        VarAutoEncoderConfig(**bad_inputs_vae)


@pytest.fixture(
    params=[
        {
            "in_shape": (1, 10, 10),
            "strides": (1, 1),
            "dropout": 0.5,
            "kernel_size": 5,
            "inter_channels": (2, 2),
            "inter_dilations": (3, 3),
            "padding": (2, 2),
        },
        {
            "in_shape": (1, 10, 10),
            "strides": ((1, 2), 1),
            "kernel_size": (3, 3),
            "padding": 2,
            "up_kernel_size": 5,
        },
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    AutoEncoderConfig(**good_inputs)
    VarAutoEncoderConfig(**good_inputs)


def test_AutoEncoderConfig():
    config = AutoEncoderConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3, 5),
        up_kernel_size=(3, 3),
        num_res_units=1,
        inter_channels=(2, 2),
        inter_dilations=(3, 3),
        num_inter_units=1,
        norm=("BATCh", {"eps": 0.1}),
        dropout=0.1,
        bias=False,
        padding=1,
    )
    assert config.spatial_dims == 2
    assert config.in_channels == 1
    assert config.out_channels == 1
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3, 5)
    assert config.num_res_units == 1
    assert config.inter_channels == (2, 2)
    assert config.inter_dilations == (3, 3)
    assert config.num_inter_units == 1
    assert config.norm == ("batch", {"eps": 0.1})
    assert config.act == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias
    assert config.padding == 1


def test_VarAutoEncoderConfig():
    config = VarAutoEncoderConfig(
        spatial_dims=2,
        in_shape=(1, 10, 10),
        out_channels=1,
        latent_size=16,
        channels=[2, 4],
        strides=[1, 1],
        kernel_size=(3, 5),
        up_kernel_size=(3, 3),
        num_res_units=1,
        inter_channels=(2, 2),
        inter_dilations=(3, 3),
        num_inter_units=1,
        norm=("BATCh", {"eps": 0.1}),
        dropout=0.1,
        bias=False,
        padding=1,
        use_sigmoid=False,
    )
    assert config.spatial_dims == 2
    assert config.in_shape == (1, 10, 10)
    assert config.out_channels == 1
    assert config.latent_size == 16
    assert config.channels == (2, 4)
    assert config.strides == (1, 1)
    assert config.kernel_size == (3, 5)
    assert config.num_res_units == 1
    assert config.inter_channels == (2, 2)
    assert config.inter_dilations == (3, 3)
    assert config.num_inter_units == 1
    assert config.norm == ("batch", {"eps": 0.1})
    assert config.act == "DefaultFromLibrary"
    assert config.dropout == 0.1
    assert not config.bias
    assert config.padding == 1
    assert not config.use_sigmoid
