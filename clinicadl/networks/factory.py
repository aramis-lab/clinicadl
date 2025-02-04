from copy import deepcopy
from typing import Any, Callable, Tuple, Union

import torch.nn as nn

import clinicadl.networks.nn as nets
from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import (
    ImplementedNetwork,
    NetworkConfig,
    NetworkType,
    create_network_config,
)
from .config.mlp_conv import ConvDecoderOptions, ConvEncoderOptions, MLPOptions
from .nn import MLP, ConvDecoder, ConvEncoder


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
    config = create_network_config(name)(**kwargs)
    _ = _update_config_with_getter(config)

    return config


def get_network_from_config(config: NetworkConfig) -> Tuple[nn.Module, NetworkConfig]:
    """
    Factory function to get a ClinicaDL neural network from a NetworkConfig instance.

    Parameters
    ----------
    config : NetworkConfig
        the configuration object.

    Returns
    -------
    nn.Module
        the neural network.
    NetworkConfig
        the updated config object: the arguments set to default will be updated
        with their effective values (the default values from the network).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    network_type = config._type  # pylint: disable=protected-access

    getter = _update_config_with_getter(config)

    if network_type == NetworkType.CUSTOM:
        config_dict = config.model_dump(exclude="name")

    else:  # sota networks
        config_dict = config.model_dump()

    network = getter(**config_dict)

    return network, config


def _update_config_with_getter(config: NetworkConfig) -> Callable:
    """
    Does the logic of parameter updates on NetworkConfig and returns
    the right getter object to instantiate the network.
    """
    network_type = config._type  # pylint: disable=protected-access

    if network_type == NetworkType.CUSTOM:
        network_class: type[nn.Module] = getattr(nets, config.name)
        if config.name == ImplementedNetwork.SE_RESNET:
            _update_config_with_defaults(
                config, getattr(nets, ImplementedNetwork.RESNET.value).__init__
            )  # SEResNet has some default values in ResNet
        elif config.name == ImplementedNetwork.ATT_UNET:
            _update_config_with_defaults(
                config, getattr(nets, ImplementedNetwork.UNET.value).__init__
            )
        _update_config_with_defaults(config, network_class.__init__)

        return network_class

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

        return getter


def _update_config_with_defaults(
    config: NetworkConfig, function: Callable
) -> NetworkConfig:
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
