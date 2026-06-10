import pytest

from clinicadl.networks.config import (
    AttentionUNetConfig,
    AutoEncoderConfig,
    CNNConfig,
    ConvDecoderConfig,
    ConvEncoderConfig,
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
    DenseNetConfig,
    GeneratorConfig,
    MLPConfig,
    ResNet18Config,
    ResNet34Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
    ResNetConfig,
    SEResNet50Config,
    SEResNet101Config,
    SEResNet152Config,
    SEResNetConfig,
    UNetConfig,
    VAEConfig,
    ViTB16Config,
    ViTB32Config,
    ViTConfig,
    ViTL16Config,
    ViTL32Config,
)
from clinicadl.networks.factory import get_network_from_dict
from clinicadl.utils.json import read_json

MANDATORY_ARGS = {
    "MLP": {"num_inputs": 1, "num_outputs": 1, "hidden_dims": [1]},
    "ConvEncoder": {"spatial_dims": 2, "in_channels": 1, "channels": [1, 2]},
    "ConvDecoder": {"spatial_dims": 2, "in_channels": 1, "channels": [1, 2]},
    "CNN": {"in_shape": (1, 3, 3), "num_outputs": 1, "conv_args": {"channels": [1]}},
    "Generator": {
        "latent_size": 1,
        "start_shape": (1, 3, 3),
        "conv_args": {"channels": [1]},
    },
    "AutoEncoder": {
        "latent_size": 1,
        "in_shape": (1, 3, 3),
        "conv_args": {"channels": [1]},
    },
    "VAE": {"latent_size": 1, "in_shape": (1, 3, 3), "conv_args": {"channels": [1]}},
    "UNet": {"spatial_dims": 2, "in_channels": 1, "out_channels": 1},
    "AttentionUNet": {"spatial_dims": 2, "in_channels": 1, "out_channels": 1},
    "DenseNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "ResNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "SEResNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "ViT": {"in_shape": (1, 3, 3), "patch_size": 1, "num_outputs": 1},
    "DenseNet121": {"num_outputs": None},
    "DenseNet161": {"num_outputs": None},
    "DenseNet169": {"num_outputs": None},
    "DenseNet201": {"num_outputs": None},
    "ResNet18": {"num_outputs": None},
    "ResNet34": {"num_outputs": None},
    "ResNet50": {"num_outputs": None},
    "ResNet101": {"num_outputs": None},
    "ResNet152": {"num_outputs": None},
    "SEResNet50": {"num_outputs": None},
    "SEResNet101": {"num_outputs": None},
    "SEResNet152": {"num_outputs": None},
    "ViTB16": {"num_outputs": None},
    "ViTB32": {"num_outputs": None},
    "ViTL16": {"num_outputs": None},
    "ViTL32": {"num_outputs": None},
}


@pytest.mark.parametrize(
    "config",
    [
        MLPConfig,
        ConvEncoderConfig,
        ConvDecoderConfig,
        CNNConfig,
        GeneratorConfig,
        AutoEncoderConfig,
        VAEConfig,
        UNetConfig,
        AttentionUNetConfig,
        DenseNetConfig,
        DenseNet121Config,
        DenseNet161Config,
        DenseNet169Config,
        DenseNet201Config,
        ResNetConfig,
        ResNet18Config,
        ResNet34Config,
        ResNet50Config,
        ResNet101Config,
        ResNet152Config,
        SEResNetConfig,
        SEResNet50Config,
        SEResNet101Config,
        SEResNet152Config,
        ViTConfig,
        ViTB16Config,
        ViTB32Config,
        ViTL16Config,
        ViTL32Config,
    ],
)
def test_get_network_from_dict(config, tmp_path):
    c = config(**MANDATORY_ARGS[config._get_name()])
    c.to_json(tmp_path / "config.json")
    config_dict = read_json(tmp_path / "config.json")
    c = get_network_from_dict(config_dict)
    assert isinstance(c, config)

    if config is UNetConfig:
        c = UNetConfig(spatial_dims=3, in_channels=5, out_channels=5)
        assert get_network_from_dict(c.to_dict()).out_channels == 5
