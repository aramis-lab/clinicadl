import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.densenet import DenseNetConfig


@pytest.fixture
def dummy_arguments():
    args = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
    }
    return args


def test_fails_validations(dummy_arguments):
    with pytest.raises(ValidationError):
        DenseNetConfig(**{**dummy_arguments, **{"dropout_prob": 1.1}})


def test_passes_validations(dummy_arguments):
    DenseNetConfig(**{**dummy_arguments, **{"dropout_prob": 0.1}})


def test_DenseNetConfig():
    config = DenseNetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_features=16,
        growth_rate=2,
        block_config=(3, 5),
        bn_size=1,
        norm=("batch", {"eps": 0.5}),
        dropout_prob=0.1,
    )
    assert config.spatial_dims == 2
    assert config.in_channels == 1
    assert config.out_channels == 1
    assert config.init_features == 16
    assert config.growth_rate == 2
    assert config.block_config == (3, 5)
    assert config.bn_size == 1
    assert config.norm == ("batch", {"eps": 0.5})
    assert config.act == "DefaultFromLibrary"
    assert config.dropout_prob
