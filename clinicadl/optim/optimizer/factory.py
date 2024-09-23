from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.optim as optim

from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import OptimizerConfig
from .utils import get_params_in_groups, get_params_not_in_groups


def get_optimizer(
    network: nn.Module,
    config: OptimizerConfig,
) -> Tuple[optim.Optimizer, OptimizerConfig]:
    """
    Factory function to get an optimizer from PyTorch.

    Parameters
    ----------
    network : nn.Module
        The neural network to optimize.
    config : OptimizerConfig
        The config class with the parameters of the optimizer.

    Returns
    -------
    optim.Optimizer
        The optimizer.
    OptimizerConfig
        The updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.

    Raises
    ------
    AttributeError
        If a parameter group mentioned in the config class cannot be found in the network.
    """
    optimizer_class = getattr(optim, config.optimizer)
    expected_args, default_args = get_args_and_defaults(optimizer_class.__init__)

    for arg, value in config.model_dump().items():
        if arg in expected_args and value != DefaultFromLibrary.YES:
            default_args[arg] = value

    args_groups, args_global = _regroup_args(default_args)

    if len(args_groups) == 0:
        list_args_groups = network.parameters()
    else:
        list_args_groups = []
        args_groups = sorted(args_groups.items())  # order in the list is important
        for group, args in args_groups:
            params, _ = get_params_in_groups(network, group)
            args.update({"params": params})
            list_args_groups.append(args)

        other_params, params_names = get_params_not_in_groups(
            network, [group for group, _ in args_groups]
        )
        if len(params_names) > 0:
            list_args_groups.append({"params": other_params})

    optimizer = optimizer_class(list_args_groups, **args_global)
    updated_config = config.model_copy(update=default_args)

    return optimizer, updated_config


def _regroup_args(
    args: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Groups arguments stored in a dict by parameter groups.

    Parameters
    ----------
    args : Dict[str, Any]
        The arguments.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        The arguments for each group.
    Dict[str, Any]
        The arguments that are common to all groups.

    Examples
    --------
    >>> args = {
            "weight_decay": {"params_0": 0.0, "params_1": 1.0},
            "alpha": {"params_1": 0.5, "ELSE": 0.1},
            "betas": (0.1, 0.1),
        }
    >>> args_groups, args_global = _regroup_args(args)
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


def _get_params_in_group(
    network: nn.Module, group: str
) -> Tuple[Iterator[torch.Tensor], List[str]]:
    """
    Gets the parameters of a specific group of a neural network.

    Parameters
    ----------
    network : nn.Module
        The neural network.
    group : str
        The name of the group, e.g. a layer or a block.
        If it is a sub-block, the hierarchy should be
        specified with "." (see examples).
        Will work even if the group is reduced to a base layer
        (e.g. group = "dense.weight" or "dense.bias").

    Returns
    -------
    Iterator[torch.Tensor]
        A generator that contains the parameters of the group.
    List[str]
        The name of all the parameters in the group.

    Raises
    ------
    AttributeError
        If `group` cannot be found in the network.

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
    >>> generator, params_names = _get_params_in_group(network, "final.dense1")
    >>> params_names
    ["final.dense1.weight", "final.dense1.bias"]
    """
    group_hierarchy = group.split(".")
    for name in group_hierarchy:
        try:
            network = getattr(network, name)
        except AttributeError as exc:
            raise AttributeError(
                f"There is no such group as {group} in the network."
            ) from exc

    try:
        params = network.parameters()
        params_names = [
            ".".join([group, name]) for name, _ in network.named_parameters()
        ]
    except AttributeError:  # we already reached params
        params = (param for param in [network])
        params_names = [group]

    return params, params_names


def _get_params_not_in_group(
    network: nn.Module, group: Iterable[str]
) -> Iterator[torch.Tensor]:
    """
    Finds the parameters of a neural networks that
    are not in a group.

    Parameters
    ----------
    network : nn.Module
        The neural network.
    group : List[str]
        The group of parameters.

    Returns
    -------
    Iterator[torch.Tensor]
        A generator of all the parameters that are not in the input
        group.
    """
    return (param[1] for param in network.named_parameters() if param[0] not in group)
