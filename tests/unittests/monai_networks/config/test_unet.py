import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.unet import AttentionUnetConfig, UNetConfig


@pytest.fixture
def dummy_arguments():
    args = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
    }
    return args


@pytest.fixture(
    params=[
        {"strides": (1, 1), "channels": (2, 4, 8), "adn_ordering": "NDB"},
        {"strides": (1, 1), "channels": (2, 4, 8), "adn_ordering": "NND"},
        {"strides": (1, 1), "channels": (2, 4, 8), "dropout": 1.1},
        {"strides": (1, 1), "channels": (2, 4, 8), "kernel_size": 4},
        {"strides": (1, 1), "channels": (2, 4, 8), "kernel_size": (3,)},
        {"strides": (1, 1), "channels": (2, 4, 8), "kernel_size": (3, 3, 3)},
        {"strides": (1, 1), "channels": (2, 4, 8), "up_kernel_size": 4},
        {"strides": (1, 1), "channels": (2, 4, 8), "up_kernel_size": (3,)},
        {"strides": (1, 1), "channels": (2, 4, 8), "up_kernel_size": (3, 3, 3)},
        {"strides": (1, 2, 3), "channels": (2, 4, 8)},
        {"strides": (1, (1, 2, 3)), "channels": (2, 4, 8)},
        {"strides": (), "channels": (2,)},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        UNetConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        AttentionUnetConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {
            "strides": (1, 1),
            "channels": (2, 4, 8),
            "adn_ordering": "DAN",
            "dropout": 0.5,
            "kernel_size": 5,
            "up_kernel_size": 5,
        },
        {
            "strides": ((1, 2),),
            "channels": (2, 4),
            "adn_ordering": "AN",
            "kernel_size": (3, 5),
            "up_kernel_size": (3, 5),
        },
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    UNetConfig(**good_inputs)
    AttentionUnetConfig(**good_inputs)


def test_UNetConfig():
    config = UNetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[2, 4],
        strides=[1],
        kernel_size=(3, 5),
        up_kernel_size=(3, 3),
        num_res_units=1,
        act="ElU",
        norm=("BATCh", {"eps": 0.1}),
        dropout=0.1,
        bias=False,
        adn_ordering="A",
    )
    assert config.spatial_dims == 2
    assert config.in_channels == 1
    assert config.out_channels == 1
    assert config.channels == (2, 4)
    assert config.strides == (1,)
    assert config.kernel_size == (3, 5)
    assert config.up_kernel_size == (3, 3)
    assert config.num_res_units == 1
    assert config.act == "elu"
    assert config.norm == ("batch", {"eps": 0.1})
    assert config.dropout == 0.1
    assert not config.bias
    assert config.adn_ordering == "A"


def test_AttentionUnetConfig():
    config = AttentionUnetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[2, 4],
        strides=[1],
        kernel_size=(3, 5),
        up_kernel_size=(3, 3),
        num_res_units=1,
        act="ElU",
        norm="inSTance",
        dropout=0.1,
        bias=False,
        adn_ordering="DA",
    )
    assert config.spatial_dims == 2
    assert config.in_channels == 1
    assert config.out_channels == 1
    assert config.channels == (2, 4)
    assert config.strides == (1,)
    assert config.kernel_size == (3, 5)
    assert config.up_kernel_size == (3, 3)
    assert config.num_res_units == 1
    assert config.act == "elu"
    assert config.norm == "instance"
    assert config.dropout == 0.1
    assert not config.bias
    assert config.adn_ordering == "DA"
