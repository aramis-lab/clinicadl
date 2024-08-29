import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.resnet import ResNetFeaturesConfig


@pytest.fixture(
    params=[
        {"model_name": "abc"},
        {"model_name": "resnet18", "pretrained": True, "spatial_dims": 2},
        {"model_name": "resnet18", "pretrained": True, "in_channels": 2},
        {
            "model_name": "resnet18",
            "in_channels": 2,
        },  # pretrained should be set to False
        {"model_name": "resnet18", "spatial_dims": 2},
    ]
)
def bad_inputs(request: pytest.FixtureRequest):
    return request.param


def test_fails_validations(bad_inputs: dict):
    with pytest.raises(ValidationError):
        ResNetFeaturesConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {"model_name": "resnet18", "pretrained": True, "spatial_dims": 3},
        {"model_name": "resnet18", "pretrained": True, "in_channels": 1},
        {"model_name": "resnet18", "pretrained": True},
        {"model_name": "resnet18", "spatial_dims": 3},
        {"model_name": "resnet18", "in_channels": 1},
    ]
)
def good_inputs(request: pytest.FixtureRequest):
    return {**request.param}


def test_passes_validations(good_inputs: dict):
    ResNetFeaturesConfig(**good_inputs)


def test_ResNetFeaturesConfig():
    config = ResNetFeaturesConfig(
        model_name="resnet200",
        pretrained=False,
        spatial_dims=2,
        in_channels=2,
    )
    assert config.network == "ResNetFeatures"
    assert config.model_name == "resnet200"
    assert not config.pretrained
    assert config.spatial_dims == 2
    assert config.in_channels == 2
