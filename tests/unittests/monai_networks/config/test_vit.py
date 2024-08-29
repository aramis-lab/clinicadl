import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.vit import (
    ViTAutoEncConfig,
    ViTConfig,
)


@pytest.fixture
def dummy_arguments():
    args = {
        "in_channels": 2,
    }
    return args


@pytest.fixture(
    params=[
        {"img_size": (16, 16, 16), "patch_size": (4, 4, 4), "dropout_rate": 1.1},
        {"img_size": (16, 16), "patch_size": 4},
        {"img_size": 16, "patch_size": (4, 4)},
        {"img_size": 16, "patch_size": (4, 4)},
        {
            "img_size": (16, 16, 16),
            "patch_size": (4, 4, 4),
            "hidden_size": 42,
            "num_heads": 5,
        },
        {"img_size": (16, 16, 16), "patch_size": (4, 4, 4), "num_heads": 5},
        {"img_size": (16, 16, 16), "patch_size": (4, 4, 4), "hidden_size": 42},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture(
    params=[
        {"img_size": (20, 20, 20), "patch_size": (4, 4, 5)},
        {"img_size": (20, 20, 20), "patch_size": (4, 4, 9)},
    ]
)
def bad_inputs_ae(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        ViTConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        ViTAutoEncConfig(**bad_inputs)


def test_fails_validations_ae(bad_inputs_ae):
    with pytest.raises(ValidationError):
        ViTAutoEncConfig(**bad_inputs_ae)


@pytest.fixture(
    params=[
        {
            "img_size": (16, 16, 16),
            "patch_size": (4, 4, 4),
            "dropout_rate": 0.5,
            "hidden_size": 42,
            "num_heads": 6,
        },
    ]
)
def good_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture(
    params=[
        {"img_size": 10, "patch_size": 3},
    ]
)
def good_inputs_vit(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


def test_passes_validations(good_inputs):
    ViTConfig(**good_inputs)
    ViTAutoEncConfig(**good_inputs)


def test_passes_validations_vit(good_inputs_vit):
    ViTConfig(**good_inputs_vit)


def test_ViTConfig():
    config = ViTConfig(
        in_channels=2,
        img_size=16,
        patch_size=4,
        hidden_size=32,
        mlp_dim=4,
        num_layers=3,
        num_heads=4,
        proj_type="perceptron",
        pos_embed_type="sincos",
        classification=True,
        num_classes=3,
        dropout_rate=0.1,
        spatial_dims=3,
        post_activation=None,
        qkv_bias=True,
    )
    assert config.network == "ViT"
    assert config.in_channels == 2
    assert config.img_size == 16
    assert config.patch_size == 4
    assert config.hidden_size == 32
    assert config.mlp_dim == 4
    assert config.num_layers == 3
    assert config.num_heads == 4
    assert config.proj_type == "perceptron"
    assert config.pos_embed_type == "sincos"
    assert config.classification
    assert config.num_classes == 3
    assert config.dropout_rate == 0.1
    assert config.spatial_dims == 3
    assert config.post_activation is None
    assert config.qkv_bias
    assert config.save_attn == "DefaultFromLibrary"


def test_ViTAutoEncConfig():
    config = ViTAutoEncConfig(
        in_channels=2,
        img_size=16,
        patch_size=4,
        out_channels=2,
        deconv_chns=7,
        hidden_size=32,
        mlp_dim=4,
        num_layers=3,
        num_heads=4,
        proj_type="perceptron",
        pos_embed_type="sincos",
        dropout_rate=0.1,
        spatial_dims=3,
        qkv_bias=True,
    )
    assert config.network == "ViTAutoEnc"
    assert config.in_channels == 2
    assert config.img_size == 16
    assert config.patch_size == 4
    assert config.out_channels == 2
    assert config.deconv_chns == 7
    assert config.hidden_size == 32
    assert config.mlp_dim == 4
    assert config.num_layers == 3
    assert config.num_heads == 4
    assert config.proj_type == "perceptron"
    assert config.pos_embed_type == "sincos"
    assert config.dropout_rate == 0.1
    assert config.spatial_dims == 3
    assert config.qkv_bias
    assert config.save_attn == "DefaultFromLibrary"
