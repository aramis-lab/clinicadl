from itertools import chain
from typing import Iterator, List, Tuple, Union

import torch
import torch.nn as nn


def get_params_in_groups(
    network: nn.Module, groups: Union[str, List[str]]
) -> Tuple[Iterator[torch.Tensor], List[str]]:
    """
    Gets the parameters of specific groups of a neural network.

    Parameters
    ----------
    network : nn.Module
        The neural network.
    groups : Union[str, List[str]]
        The name of the group(s), e.g. a layer or a block.
        If the user refers to a sub-block, the hierarchy should be
        specified with "." (see examples).
        If a list is passed, the function will output the parameters
        of all groups mentioned together.

    Returns
    -------
    Iterator[torch.Tensor]
        An iterator that contains the parameters of the group(s).
    List[str]
        The name of all the parameters in the group(s).

    Examples
    --------
    >>> net = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 1, kernel_size=3)),
                    ("final", nn.Sequential(OrderedDict([("dense1", nn.Linear(10, 10))]))),
                ]
            )
        )
    >>> params, params_names = get_params_in_groups(network, "final.dense1")
    >>> params_names
    ["final.dense1.weight", "final.dense1.bias"]
    >>> params, params_names = get_params_in_groups(network, ["conv1.weight", "final"])
    >>> params_names
    ["conv1.weight", "final.dense1.weight", "final.dense1.bias"]
    """
    if isinstance(groups, str):
        groups = [groups]

    params = iter(())
    params_names = []
    for group in groups:
        network_ = network
        group_hierarchy = group.split(".")
        for name in group_hierarchy:
            network_ = getattr(network_, name)

        try:
            params = chain(params, network_.parameters())
            params_names += [
                ".".join([group, name]) for name, _ in network_.named_parameters()
            ]
        except AttributeError:  # we already reached params
            params = chain(params, (param for param in [network_]))
            params_names += [group]

    return params, params_names


def get_params_not_in_groups(
    network: nn.Module, groups: Union[str, List[str]]
) -> Tuple[Iterator[torch.Tensor], List[str]]:
    """
    Gets the parameters not in specific groups of a neural network.

    Parameters
    ----------
    network : nn.Module
        The neural network.
    groups : Union[str, List[str]]
        The name of the group(s), e.g. a layer or a block.
        If the user refers to a sub-block, the hierarchy should be
        specified with "." (see examples).
        If a list is passed, the function will output the parameters
        that are not in any group of that list.

    Returns
    -------
    Iterator[torch.Tensor]
        An iterator that contains the parameters not in the group(s).
    List[str]
        The name of all the parameters not in the group(s).

    Examples
    --------
    >>> net = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 1, kernel_size=3)),
                    ("final", nn.Sequential(OrderedDict([("dense1", nn.Linear(10, 10))]))),
                ]
            )
        )
    >>> params, params_names = get_params_in_groups(network, "final")
    >>> params_names
    ["conv1.weight", "conv1.bias"]
    >>> params, params_names = get_params_in_groups(network, ["conv1.bias", "final"])
    >>> params_names
    ["conv1.weight"]
    """
    _, in_groups = get_params_in_groups(network, groups)
    params = (
        param[1] for param in network.named_parameters() if param[0] not in in_groups
    )
    params_names = list(
        param[0] for param in network.named_parameters() if param[0] not in in_groups
    )
    return params, params_names
