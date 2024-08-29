import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.resnet import ResNetConfig


@pytest.fixture
def dummy_arguments():
    args = {
        "block": "basic",
        "layers": (2, 2, 2, 2),
    }
    return args


@pytest.fixture(
    params=[
        {"block_inplanes": (2, 4, 8)},
        {"block_inplanes": (2, 4, 8, 16), "conv1_t_size": (3, 3)},
        {"block_inplanes": (2, 4, 8, 16), "conv1_t_stride": (3, 3)},
        {"block_inplanes": (2, 4, 8, 16), "shortcut_type": "C"},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        ResNetConfig(**bad_inputs)


@pytest.fixture(
    params=[
        {
            "block_inplanes": (2, 4, 8, 16),
            "conv1_t_size": (3, 3, 3),
            "conv1_t_stride": (3, 3, 3),
            "shortcut_type": "B",
        },
        {"block_inplanes": (2, 4, 8, 16), "conv1_t_size": 3, "conv1_t_stride": 3},
    ]
)
def good_inputs(request: pytest.FixtureRequest, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    ResNetConfig(**good_inputs)


def test_ResNetConfig():
    config = ResNetConfig(
        block="bottleneck",
        layers=(2, 2, 2, 2),
        block_inplanes=(2, 4, 8, 16),
        spatial_dims=3,
        n_input_channels=3,
        conv1_t_size=3,
        conv1_t_stride=4,
        no_max_pool=True,
        shortcut_type="A",
        widen_factor=0.8,
        num_classes=3,
        feed_forward=False,
        bias_downsample=False,
        act=("relu", {"inplace": False}),
    )
    assert config.network == "ResNet"
    assert config.block == "bottleneck"
    assert config.layers == (2, 2, 2, 2)
    assert config.block_inplanes == (2, 4, 8, 16)
    assert config.spatial_dims == 3
    assert config.n_input_channels == 3
    assert config.conv1_t_size == 3
    assert config.conv1_t_stride == 4
    assert config.no_max_pool
    assert config.shortcut_type == "A"
    assert config.widen_factor == 0.8
    assert config.num_classes == 3
    assert not config.feed_forward
    assert not config.bias_downsample
    assert config.act == ("relu", {"inplace": False})
