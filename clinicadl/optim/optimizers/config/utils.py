from itertools import chain
from typing import Any, Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "get_params_in_groups",
    "get_params_not_in_groups",
    "regroup_args_by_param_group",
]


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


def regroup_args_by_param_group(
    args: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Groups arguments stored in a dict by parameter groups.

    Parameters
    ----------
    args : Dict[str, Any]
        the arguments.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        the arguments for each group.
    Dict[str, Any]
        the arguments that are common to all groups.

    Examples
    --------
    >>> args = {
            "weight_decay": {"params_0": 0.0, "params_1": 1.0},
            "alpha": {"params_1": 0.5, "ELSE": 0.1},
            "betas": (0.1, 0.1),
        }
    >>> args_groups, args_global = _regroup_args_by_param_group(args)
    >>> args_groups
    {
        "params_0": {"weight_decay": 0.0},
        "params_1": {"alpha": 0.5, "weight_decay": 1.0},
    }
    >>> args_global
        {"betas": (0.1, 0.1), "alpha": 0.1}

    Notes
    -----
    "ELSE" is a special keyword. Passed as a group, it
    enables the user to give a value for the rest of the
    parameters (see examples).
    """
    args_groups = {}
    args_global = {}
    for arg, value in args.items():
        if isinstance(value, dict):
            for group, v in value.items():
                if group == "ELSE":
                    args_global[arg] = v
                else:
                    try:
                        args_groups[group][arg] = v
                    except KeyError:  # the first time this group is seen
                        args_groups[group] = {arg: v}
        else:
            args_global[arg] = value

    return args_groups, args_global
