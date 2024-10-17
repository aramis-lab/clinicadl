from typing import Type, Union

# pylint: disable=unused-import
from .autoencoder import AutoEncoderConfig, VAEConfig
from .base import ImplementedNetworks, NetworkConfig
from .cnn import CNNConfig
from .conv_decoder import ConvDecoderConfig
from .conv_encoder import ConvEncoderConfig
from .densenet import (
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
    DenseNetConfig,
)
from .generator import GeneratorConfig
from .mlp import MLPConfig
from .resnet import (
    ResNet18Config,
    ResNet34Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
    ResNetConfig,
)
from .senet import (
    SEResNet50Config,
    SEResNet101Config,
    SEResNet152Config,
    SEResNetConfig,
)
from .unet import AttentionUNetConfig, UNetConfig
from .vit import ViTConfig


def create_network_config(
    network: Union[str, ImplementedNetworks],
) -> Type[NetworkConfig]:
    """
    A factory function to create a config class suited for the network.

    Parameters
    ----------
    network : Union[str, ImplementedNetworks]
        The name of the neural network.

    Returns
    -------
    Type[NetworkConfig]
        The config class.
    """
    network = ImplementedNetworks(network).value.replace("-", "").replace("/", "")
    config_name = "".join([network, "Config"])
    config = globals()[config_name]

    return config
