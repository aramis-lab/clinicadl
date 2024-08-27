from enum import Enum
from typing import Type, Union

from .autoencoder import *
from .base import NetworkBaseConfig
from .classifier import *
from .densenet import *
from .fcn import *
from .generator import *
from .regressor import *

# from .resnet import *
from .unet import *
from .vit import *


class ImplementedNetworks(str, Enum):
    """Implemented neural networks in ClinicaDL."""

    REGRESSOR = "Regressor"
    CLASSIFIER = "Classifier"
    DISCRIMINATOR = "Discriminator"
    CRITIC = "Critic"
    AUTO_ENCODER = "AutoEncoder"
    VAR_AUTO_ENCODER = "VarAutoEncoder"
    DENSE_NET = "DenseNet"
    FCN = "FullyConnectedNet"
    VAR_FCN = "VarFullyConnected"
    GENERATOR = "Generator"
    RES_NET = "ResNet"
    RES_NET_FEATURES = "ResNetFeatures"
    SEG_RES_NET = "SegResNet"
    UNET = "UNet"
    ATT_UNET = "AttentionUnet"
    VIT = "ViT"
    VIT_AUTO_ENC = "ViTAutoEnc"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented neural networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )


def create_training_config(
    network: Union[str, ImplementedNetworks],
) -> Type[NetworkBaseConfig]:
    """
    A factory function to create a config class suited for the network.

    Parameters
    ----------
    network : Union[str, ImplementedNetworks]
        THe name of the neural network.

    Returns
    -------
    Type[NetworkBaseConfig]
        The config class.
    """
    network = ImplementedNetworks(network)
    config_name = "".join([network, "Config"])
    config = globals()[config_name]

    return config
