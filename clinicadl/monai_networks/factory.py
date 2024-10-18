from copy import deepcopy
from typing import Any, Callable, Tuple, Union

import torch.nn as nn
from pydantic import BaseModel

import clinicadl.monai_networks.nn as nets
from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import (
    ImplementedNetworks,
    NetworkConfig,
    NetworkType,
    create_network_config,
)
from .config.conv_decoder import ConvDecoderOptions
from .config.conv_encoder import ConvEncoderOptions
from .config.mlp import MLPOptions
from .nn import MLP, ConvDecoder, ConvEncoder


def get_network(
    name: Union[str, ImplementedNetworks], return_config: bool = False, **kwargs: Any
) -> Union[nn.Module, Tuple[nn.Module, NetworkConfig]]:
    """
    Factory function to get a neural network from its name and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedNetworks]
        the name of the neural network. Check our documentation to know
        available networks.
    return_config : bool (optional, default=False)
        if the function should return the config class regrouping the parameters of the
        neural network. Useful to keep track of the hyperparameters.
    kwargs : Any
        the parameters of the neural network. Check our documentation on networks to
        know these parameters.

    Returns
    -------
    nnn.Module
        the neural network.
    NetworkConfig
        the associated config class. Only returned if `return_config` is True.
    """
    config = create_network_config(name)(**kwargs)
    network, updated_config = get_network_from_config(config)

    return network if not return_config else (network, updated_config)


def get_network_from_config(config: NetworkConfig) -> Tuple[nn.Module, NetworkConfig]:
    """
    Factory function to get a neural network from a NetworkConfig instance.

    Parameters
    ----------
    config : NetworkConfig
        the configuration object.

    Returns
    -------
    nn.Module
        the neural network.
    NetworkConfig
        the updated config class: the arguments set to default will be updated
        with their effective values (the default values from the network).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    network_type = config._type  # pylint: disable=protected-access

    if network_type == NetworkType.CUSTOM:
        network_class: type[nn.Module] = getattr(nets, config.name)
        if config.name == ImplementedNetworks.SE_RESNET:
            _update_config_with_defaults(
                config, getattr(nets, ImplementedNetworks.RESNET.value).__init__
            )  # SEResNet has some default values in ResNet
        elif config.name == ImplementedNetworks.ATT_UNET:
            _update_config_with_defaults(
                config, getattr(nets, ImplementedNetworks.UNET.value).__init__
            )
        _update_config_with_defaults(config, network_class.__init__)

        config_dict = config.model_dump(exclude={"name", "_type"})
        network = network_class(**config_dict)

    else:  # sota networks
        if network_type == NetworkType.RESNET:
            getter: Callable[..., nn.Module] = nets.get_resnet
        elif network_type == NetworkType.DENSENET:
            getter: Callable[..., nn.Module] = nets.get_densenet
        elif network_type == NetworkType.SE_RESNET:
            getter: Callable[..., nn.Module] = nets.get_seresnet
        elif network_type == NetworkType.VIT:
            getter: Callable[..., nn.Module] = nets.get_vit
        _update_config_with_defaults(config, getter)  # pylint: disable=possibly-used-before-assignment

        config_dict = config.model_dump(exclude={"_type"})
        network = getter(**config_dict)

    return network, config


def _update_config_with_defaults(config: BaseModel, function: Callable) -> BaseModel:
    """
    Updates a config object by setting the parameters left to 'default' to their actual
    default values, extracted from 'function'.
    """
    _, defaults = get_args_and_defaults(function)

    for arg, value in config:
        if isinstance(value, MLPOptions):
            _update_config_with_defaults(
                value, MLP.__init__
            )  # we need to update the sub config object
        elif isinstance(value, ConvEncoderOptions):
            _update_config_with_defaults(value, ConvEncoder.__init__)
        elif isinstance(value, ConvDecoderOptions):
            _update_config_with_defaults(value, ConvDecoder.__init__)
        elif value == DefaultFromLibrary.YES and arg in defaults:
            setattr(config, arg, defaults[arg])
