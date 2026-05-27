from copy import deepcopy

import pytest

import clinicadl.networks.nn as nets
from clinicadl.networks.config import *

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
    "config,network",
    [
        (MLPConfig, nets.MLP),
        (ConvEncoderConfig, nets.ConvEncoder),
        (ConvDecoderConfig, nets.ConvDecoder),
        (CNNConfig, nets.CNN),
        (GeneratorConfig, nets.Generator),
        (AutoEncoderConfig, nets.AutoEncoder),
        (VAEConfig, nets.VAE),
        (DenseNetConfig, nets.DenseNet),
        (ResNetConfig, nets.ResNet),
        (SEResNetConfig, nets.SEResNet),
        (UNetConfig, nets.UNet),
        (AttentionUNetConfig, nets.AttentionUNet),
        (ViTConfig, nets.ViT),
        (DenseNet121Config, nets.DenseNet),
        (DenseNet161Config, nets.DenseNet),
        (DenseNet169Config, nets.DenseNet),
        (DenseNet201Config, nets.DenseNet),
        (ResNet18Config, nets.ResNet),
        (ResNet34Config, nets.ResNet),
        (ResNet50Config, nets.ResNet),
        (ResNet101Config, nets.ResNet),
        (ResNet152Config, nets.ResNet),
        (SEResNet50Config, nets.SEResNet50),
        (SEResNet101Config, nets.SEResNet101),
        (SEResNet152Config, nets.SEResNet152),
        (ViTB16Config, nets.ViT),
        (ViTB32Config, nets.ViT),
        (ViTL16Config, nets.ViT),
        (ViTL32Config, nets.ViT),
    ],
)
def test_get_object(config, network):
    name = config._get_name()
    args = deepcopy(MANDATORY_ARGS[name])
    c = config(**args)
    network_from_config = c.get_object()
    assert isinstance(network_from_config, network)

    if name == "AutoEncoder":
        config = AutoEncoderConfig(
            latent_size=1,
            in_shape=(1, 10, 10),
            conv_args={"channels": [1, 2], "dropout": 0.2},
            mlp_args={"hidden_dims": [5], "act": "relu"},
        )
        net = config.get_object()
        assert isinstance(net, nets.AutoEncoder)
        assert net.encoder.mlp.out_channels == 1
        assert net.encoder.mlp.config.hidden_dims == [5]
        assert net.encoder.mlp.act == "relu"
        assert net.encoder.mlp.config.norm == "batch"
        assert net.config.in_shape == (1, 10, 10)
        assert net.encoder.convolutions.config.channels == [1, 2]
        assert net.encoder.convolutions.config.dropout == 0.2
        assert net.encoder.convolutions.config.act == "prelu"


def test_name():
    for name in ImplementedNetwork:
        config = globals()[f"{name.value}Config"]
        c = config(**MANDATORY_ARGS[name])
        assert c.name_ == name.value
