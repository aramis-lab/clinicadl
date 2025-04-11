import pytest

from clinicadl.networks.config import get_network_config
from clinicadl.networks.config.cnns import (
    AutoEncoderConfig,
    CNNConfig,
    GeneratorConfig,
    VAEConfig,
)
from clinicadl.networks.config.densenet import (
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
    DenseNetConfig,
)
from clinicadl.networks.config.mlp_conv import (
    ConvDecoderConfig,
    ConvEncoderConfig,
    MLPConfig,
)
from clinicadl.networks.config.resnet import (
    ResNet18Config,
    ResNet34Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
    ResNetConfig,
)
from clinicadl.networks.config.senet import (
    SEResNet50Config,
    SEResNet101Config,
    SEResNet152Config,
    SEResNetConfig,
)
from clinicadl.networks.config.unet import AttentionUNetConfig, UNetConfig
from clinicadl.networks.config.vit import (
    ViTB16Config,
    ViTB32Config,
    ViTConfig,
    ViTL16Config,
    ViTL32Config,
)


@pytest.mark.parametrize(
    "args,name,config",
    [
        (
            {
                "num_outputs": 1,
                "num_inputs": 1,
                "hidden_dims": [1],
            },
            "MLP",
            MLPConfig,
        ),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "channels": [1, 1],
            },
            "ConvDecoder",
            ConvDecoderConfig,
        ),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "channels": [1, 1],
            },
            "ConvEncoder",
            ConvEncoderConfig,
        ),
        (
            {
                "in_shape": (1, 6, 6),
                "conv_args": {"channels": [1]},
                "num_outputs": 1,
            },
            "CNN",
            CNNConfig,
        ),
        (
            {
                "start_shape": (1, 4, 4),
                "conv_args": {"channels": [1]},
                "latent_size": 1,
            },
            "Generator",
            GeneratorConfig,
        ),
        (
            {
                "in_shape": (1, 6, 6),
                "conv_args": {"channels": [1]},
                "latent_size": 1,
            },
            "AutoEncoder",
            AutoEncoderConfig,
        ),
        (
            {
                "in_shape": (1, 6, 6),
                "conv_args": {"channels": [1]},
                "latent_size": 1,
            },
            "VAE",
            VAEConfig,
        ),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
            "DenseNet",
            DenseNetConfig,
        ),
        ({"num_outputs": None}, "DenseNet-201", DenseNet201Config),
        ({"num_outputs": None}, "DenseNet-121", DenseNet121Config),
        ({"num_outputs": None}, "DenseNet-161", DenseNet161Config),
        ({"num_outputs": None}, "DenseNet-169", DenseNet169Config),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
            "ResNet",
            ResNetConfig,
        ),
        ({"num_outputs": None}, "ResNet-101", ResNet101Config),
        ({"num_outputs": None}, "ResNet-152", ResNet152Config),
        ({"num_outputs": None}, "ResNet-18", ResNet18Config),
        ({"num_outputs": None}, "ResNet-34", ResNet34Config),
        ({"num_outputs": None}, "ResNet-50", ResNet50Config),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
            "SEResNet",
            SEResNetConfig,
        ),
        ({"num_outputs": None}, "SEResNet-101", SEResNet101Config),
        ({"num_outputs": None}, "SEResNet-152", SEResNet152Config),
        ({"num_outputs": None}, "SEResNet-50", SEResNet50Config),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 1,
            },
            "UNet",
            UNetConfig,
        ),
        (
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 1,
            },
            "AttentionUNet",
            AttentionUNetConfig,
        ),
        (
            {
                "in_shape": (1, 6, 6),
                "num_outputs": 1,
                "patch_size": 3,
            },
            "ViT",
            ViTConfig,
        ),
        ({"num_outputs": None}, "ViT-B/16", ViTB16Config),
        ({"num_outputs": None}, "ViT-B/32", ViTB32Config),
        ({"num_outputs": None}, "ViT-L/16", ViTL16Config),
        ({"num_outputs": None}, "ViT-L/32", ViTL32Config),
    ],
)
def test_get_network_config(args, name, config):
    c = get_network_config(name, **args)
    assert c.name == name
    assert isinstance(c, config)

    if name == "CNN":
        config = get_network_config(
            "CNN",
            in_shape=(1, 6, 6),
            conv_args={"channels": [1]},
            num_outputs=1,
            mlp_args={"dropout": 1, "hidden_dims": [1]},
        )
        assert config.name == "CNN"
        assert config.in_shape == (1, 6, 6)
        assert config.conv_args.channels == [1]
        assert config.num_outputs == 1
        assert config.mlp_args.dropout == 1

        with pytest.raises(ValueError):
            get_network_config("abc")
