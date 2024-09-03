from typing import Tuple

import monai.networks.nets as networks
import torch.nn as nn

from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config.base import NetworkConfig


def get_network(config: NetworkConfig) -> Tuple[nn.Module, NetworkConfig]:
    """
    Factory function to get a Neural Network from MONAI.

    Parameters
    ----------
    config : NetworkConfig
        The config class with the parameters of the network.

    Returns
    -------
    nn.Module
        The neural network.
    NetworkConfig
        The updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    network_class = getattr(networks, config.network)
    expected_args, config_dict = get_args_and_defaults(network_class.__init__)
    for arg, value in config.model_dump().items():
        if arg in expected_args and value != DefaultFromLibrary.YES:
            config_dict[arg] = value

    network = network_class(**config_dict)
    updated_config = config.model_copy(update=config_dict)

    return network, updated_config
