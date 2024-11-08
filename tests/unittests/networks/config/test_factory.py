import pytest

from clinicadl.networks.config import create_network_config
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
    "name,expected_class",
    [
        ("MLP", MLPConfig),
        ("ConvDecoder", ConvDecoderConfig),
        ("ConvEncoder", ConvEncoderConfig),
        ("CNN", CNNConfig),
        ("Generator", GeneratorConfig),
        ("AutoEncoder", AutoEncoderConfig),
        ("VAE", VAEConfig),
        ("DenseNet", DenseNetConfig),
        ("DenseNet-201", DenseNet201Config),
        ("DenseNet-121", DenseNet121Config),
        ("DenseNet-161", DenseNet161Config),
        ("DenseNet-169", DenseNet169Config),
        ("ResNet-101", ResNet101Config),
        ("ResNet-152", ResNet152Config),
        ("ResNet-18", ResNet18Config),
        ("ResNet-34", ResNet34Config),
        ("ResNet-50", ResNet50Config),
        ("ResNet", ResNetConfig),
        ("SEResNet-101", SEResNet101Config),
        ("SEResNet-152", SEResNet152Config),
        ("SEResNet-50", SEResNet50Config),
        ("SEResNet", SEResNetConfig),
        ("UNet", UNetConfig),
        ("AttentionUNet", AttentionUNetConfig),
        ("ViT-B/16", ViTB16Config),
        ("ViT-B/32", ViTB32Config),
        ("ViT", ViTConfig),
        ("ViT-L/16", ViTL16Config),
        ("ViT-L/32", ViTL32Config),
    ],
)
def test_create_optimizer_config(name, expected_class):
    config = create_network_config(name)
    assert config == expected_class
