from typing import Any, Callable, Tuple, Union

import torch.nn as nn

import clinicadl.monai_networks.nn as nets
from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import (
    ImplementedNetworks,
    NetworkConfig,
    NetworkType,
    create_network_config,
)


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
    Union[nn.Module, Tuple[nn.Module, NetworkConfig]]
        the neural network, and the associated config class if `return_config` is True.
    """
    config = create_network_config(name)(**kwargs)
    network_type = config._type  # pylint: disable=protected-access

    if network_type == NetworkType.CUSTOM:
        getter: type[nn.Module] = getattr(nets, config.name)
        _, config_dict = get_args_and_defaults(getter.__init__)
    else:  # sota networks
        if network_type == NetworkType.RESNET:
            getter: Callable[..., nn.Module] = nets.get_resnet
        elif network_type == NetworkType.DENSENET:
            getter: Callable[..., nn.Module] = nets.get_densenet
        elif network_type == NetworkType.SE_RESNET:
            getter: Callable[..., nn.Module] = nets.get_seresnet
        elif network_type == NetworkType.VIT:
            getter: Callable[..., nn.Module] = nets.get_vit
        _, config_dict = get_args_and_defaults(getter)

    for arg, value in config.model_dump().items():
        if value != DefaultFromLibrary.YES:  # update config with defaults
            config_dict[arg] = value

    network = getter(**config_dict)
    updated_config = config.model_copy(update=config_dict)

    return network if not return_config else (network, updated_config)
