import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.fcn import (
    FullyConnectedNetConfig,
    VarFullyConnectedNetConfig,
)


@pytest.fixture
def dummy_arguments():
    args = {
        "in_channels": 5,
        "out_channels": 1,
        "hidden_channels": [3, 2],
        "latent_size": 16,
        "encode_channels": [2, 3],
        "decode_channels": [3, 2],
    }
    return args


@pytest.fixture(
    params=[
        {"dropout": 1.1},
        {"adn_ordering": "NDB"},
        {"adn_ordering": "NND"},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        FullyConnectedNetConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        VarFullyConnectedNetConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {"dropout": 0.5, "adn_ordering": "DAN"},
        {"adn_ordering": "AN"},
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    FullyConnectedNetConfig(**good_inputs)
    VarFullyConnectedNetConfig(**good_inputs)


def test_FullyConnectedNetConfig():
    config = FullyConnectedNetConfig(
        in_channels=5,
        out_channels=1,
        hidden_channels=[3, 2],
        dropout=None,
        act="prelu",
        bias=False,
        adn_ordering="ADN",
    )
    assert config.network == "FullyConnectedNet"
    assert config.in_channels == 5
    assert config.out_channels == 1
    assert config.hidden_channels == (3, 2)
    assert config.dropout is None
    assert config.act == "prelu"
    assert not config.bias
    assert config.adn_ordering == "ADN"


def test_VarFullyConnectedNetConfig():
    config = VarFullyConnectedNetConfig(
        in_channels=5,
        out_channels=1,
        latent_size=16,
        encode_channels=[2, 3],
        decode_channels=[3, 2],
        dropout=0.1,
        act="prelu",
        bias=False,
        adn_ordering="ADN",
    )
    assert config.network == "VarFullyConnectedNet"
    assert config.in_channels == 5
    assert config.out_channels == 1
    assert config.latent_size == 16
    assert config.encode_channels == (2, 3)
    assert config.decode_channels == (3, 2)
    assert config.dropout == 0.1
    assert config.act == "prelu"
    assert not config.bias
    assert config.adn_ordering == "ADN"
