from typing import Type, Union

from .autoencoder import *
from .base import NetworkConfig
from .classifier import *
from .densenet import *
from .fcn import *
from .generator import *
from .regressor import *
from .resnet import *
from .unet import *
from .utils.enum import ImplementedNetworks
from .vit import *


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
    network = ImplementedNetworks(network)
    config_name = "".join([network, "Config"])
    config = globals()[config_name]

    return config
