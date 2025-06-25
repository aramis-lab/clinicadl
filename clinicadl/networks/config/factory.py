from typing import Any, Union

from .base import ImplementedNetwork, NetworkConfig

# pylint: disable=unused-import
from .cnns import AutoEncoderConfig, CNNConfig, GeneratorConfig, VAEConfig
from .densenet import (
    DenseNet121Config,
    DenseNet161Config,
    DenseNet169Config,
    DenseNet201Config,
    DenseNetConfig,
)
from .mlp_conv import ConvDecoderConfig, ConvEncoderConfig, MLPConfig
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
from .vit import ViTB16Config, ViTB32Config, ViTConfig, ViTL16Config, ViTL32Config


def get_network_config(
    name: Union[str, ImplementedNetwork], **kwargs: Any
) -> NetworkConfig:
    """
    Factory function to get a network configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedNetwork]
        the name of the network. Check our documentation to know
        available networks.
    **kwargs : Any
        any parameter of the network. Check our documentation on networks to
        know these parameters.

    Returns
    -------
    NetworkConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    network = ImplementedNetwork(name).value
    config_name = f"{network}Config"
    config = globals()[config_name]

    return config(**kwargs)
