from copy import deepcopy

import pytest
from pydantic import ValidationError

import clinicadl.networks.nn as nets
from clinicadl.networks.config import *
from clinicadl.networks.nn.conv_decoder import ConvDecoderOptions
from clinicadl.networks.nn.conv_encoder import ConvEncoderOptions
from clinicadl.networks.nn.layers.utils import ActFunction
from clinicadl.networks.nn.mlp import MLPOptions


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
        assert net.encoder.mlp.hidden_dims == [5]
        assert net.encoder.mlp.act == "relu"
        assert net.encoder.mlp.norm == "batch"
        assert net.in_shape == (1, 10, 10)
        assert net.encoder.convolutions.channels == [1, 2]
        assert net.encoder.convolutions.dropout == 0.2
        assert net.encoder.convolutions.act == "prelu"


def test_name():
    for name in ImplementedNetwork:
        config = globals()[f"{name.value}Config"]
    c = config(**MANDATORY_ARGS)
    assert c.name == name.value
