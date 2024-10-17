import pytest

from clinicadl.monai_networks.config.densenet import (
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
)
from clinicadl.monai_networks.config.resnet import (
    ResNet18Config,
    ResNet34Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
)
from clinicadl.monai_networks.config.senet import (
    SEResNet50Config,
    SEResNet101Config,
    SEResNet152Config,
)
from clinicadl.monai_networks.config.vit import (
    ViTB16Config,
    ViTB32Config,
    ViTL16Config,
    ViTL32Config,
)


@pytest.mark.parametrize(
    "config_class",
    [DenseNet121Config, DenseNet161Config, DenseNet169Config, DenseNet201Config],
)
def test_sota_densenet_config(config_class):
    config = config_class(pretrained=True, num_outputs=None)

    assert config.num_outputs is None
    assert config.pretrained
    assert config.output_act == "DefaultFromLibrary"
    assert config._type == "sota-DenseNet"


@pytest.mark.parametrize(
    "config_class",
    [ResNet18Config, ResNet34Config, ResNet50Config, ResNet101Config, ResNet152Config],
)
def test_sota_resnet_config(config_class):
    config = config_class(pretrained=False, num_outputs=None)

    assert config.num_outputs is None
    assert not config.pretrained
    assert config.output_act == "DefaultFromLibrary"
    assert config._type == "sota-ResNet"


@pytest.mark.parametrize(
    "config_class", [SEResNet50Config, SEResNet101Config, SEResNet152Config]
)
def test_sota_senet_config(config_class):
    config = config_class(output_act="relu", num_outputs=1)

    assert config.num_outputs == 1
    assert config.pretrained == "DefaultFromLibrary"
    assert config.output_act == "relu"
    assert config._type == "sota-SEResNet"


@pytest.mark.parametrize(
    "config_class", [ViTB16Config, ViTB32Config, ViTL16Config, ViTL32Config]
)
def test_sota_vit_config(config_class):
    config = config_class(output_act="relu", num_outputs=1)

    assert config.num_outputs == 1
    assert config.pretrained == "DefaultFromLibrary"
    assert config.output_act == "relu"
    assert config._type == "sota-ViT"


def test_autoencoder_config():
    from clinicadl.monai_networks.config.autoencoder import AutoEncoderConfig

    config = AutoEncoderConfig(
        in_shape=(1, 10, 10),
        latent_size=1,
        conv_args={"channels": [1]},
        output_act="softmax",
    )
    assert config.in_shape == (1, 10, 10)
    assert config.conv_args.channels == [1]
    assert config.output_act == "softmax"
    assert config.out_channels == "DefaultFromLibrary"


def test_vae_config():
    from clinicadl.monai_networks.config.autoencoder import VAEConfig

    config = VAEConfig(
        in_shape=(1, 10),
        latent_size=1,
        conv_args={"channels": [1], "adn_ordering": "NA"},
        output_act=("elu", {"alpha": 0.1}),
    )
    assert config.in_shape == (1, 10)
    assert config.conv_args.adn_ordering == "NA"
    assert config.output_act == ("elu", {"alpha": 0.1})
    assert config.mlp_args == "DefaultFromLibrary"


def test_cnn_config():
    from clinicadl.monai_networks.config.cnn import CNNConfig

    config = CNNConfig(
        in_shape=(2, 10, 10, 10), num_outputs=1, conv_args={"channels": [1]}
    )
    assert config.in_shape == (2, 10, 10, 10)
    assert config.conv_args.channels == [1]
    assert config.mlp_args == "DefaultFromLibrary"


def test_conv_decoder_config():
    from clinicadl.monai_networks.config.conv_decoder import ConvDecoderConfig

    config = ConvDecoderConfig(
        in_channels=1, spatial_dims=2, channels=[1, 2], kernel_size=(3, 4)
    )
    assert config.in_channels == 1
    assert config.kernel_size == (3, 4)
    assert config.stride == "DefaultFromLibrary"


def test_conv_encoder_config():
    from clinicadl.monai_networks.config.conv_encoder import ConvEncoderConfig

    config = ConvEncoderConfig(
        in_channels=1, spatial_dims=2, channels=[1, 2], kernel_size=[(3, 4), (4, 5)]
    )
    assert config.in_channels == 1
    assert config.kernel_size == [(3, 4), (4, 5)]
    assert config.padding == "DefaultFromLibrary"


def test_mlp_config():
    from clinicadl.monai_networks.config.mlp import MLPConfig

    config = MLPConfig(
        in_channels=1, out_channels=1, hidden_channels=[2, 3], dropout=0.1
    )
    assert config.in_channels == 1
    assert config.dropout == 0.1
    assert config.act == "DefaultFromLibrary"


def test_resnet_config():
    from clinicadl.monai_networks.config.resnet import ResNetConfig

    config = ResNetConfig(
        spatial_dims=1, in_channels=1, num_outputs=None, block_type="bottleneck"
    )
    assert config.num_outputs is None
    assert config.block_type == "bottleneck"
    assert config.bottleneck_reduction == "DefaultFromLibrary"


def test_seresnet_config():
    from clinicadl.monai_networks.config.senet import SEResNetConfig

    config = SEResNetConfig(
        spatial_dims=1,
        in_channels=1,
        num_outputs=None,
        block_type="bottleneck",
        se_reduction=2,
    )
    assert config.num_outputs is None
    assert config.block_type == "bottleneck"
    assert config.se_reduction == 2
    assert config.bottleneck_reduction == "DefaultFromLibrary"


def test_densenet_config():
    from clinicadl.monai_networks.config.densenet import DenseNetConfig

    config = DenseNetConfig(
        spatial_dims=1, in_channels=1, num_outputs=2, n_dense_layers=(1, 2)
    )
    assert config.num_outputs == 2
    assert config.n_dense_layers == (1, 2)
    assert config.growth_rate == "DefaultFromLibrary"


def test_vit_config():
    from clinicadl.monai_networks.config.vit import ViTConfig

    config = ViTConfig(in_shape=(1, 10), patch_size=2, num_outputs=1, embedding_dim=42)
    assert config.num_outputs == 1
    assert config.embedding_dim == 42
    assert config.mlp_dim == "DefaultFromLibrary"


def test_unet_config():
    from clinicadl.monai_networks.config.unet import UNetConfig

    config = UNetConfig(spatial_dims=1, in_channels=1, out_channels=1, channels=(4, 8))
    assert config.out_channels == 1
    assert config.channels == (4, 8)
    assert config.output_act == "DefaultFromLibrary"


def test_att_unet_config():
    from clinicadl.monai_networks.config.unet import AttentionUNetConfig

    config = AttentionUNetConfig(
        spatial_dims=1,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        output_act="softmax",
    )
    assert config.spatial_dims == 1
    assert config.output_act == "softmax"
    assert config.dropout == "DefaultFromLibrary"
