from copy import deepcopy
from typing import Any, Dict, Tuple, Union

import torch.nn as nn
import torch.optim as optim

from clinicadl.utils.factories import update_config_with_defaults

from .config import ImplementedOptimizer, OptimizerConfig, create_optimizer_config
from .utils import get_params_in_groups, get_params_not_in_groups


def get_optimizer_config(
    name: Union[str, ImplementedOptimizer],
    **kwargs: Any,
) -> OptimizerConfig:
    """
    Factory function to get an optimizer configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedOptimizer]
        the name of the optimizer. Check our documentation to know
        available optimizers.
    **kwargs : Any
        any parameter of the optimizer. Check our documentation on optimizers to
        know these parameters.

    Returns
    -------
    OptimizerConfig
        the configuration object. Default values will be returned for the parameters
        not passed by the user.
    """
    config = create_optimizer_config(name)(**kwargs)
    optimizer_class = getattr(optim, config.name)

    update_config_with_defaults(config, function=optimizer_class.__init__)

    return config


def get_optimizer_from_config(
    config: OptimizerConfig,
    network: nn.Module,
) -> Tuple[optim.Optimizer, OptimizerConfig]:
    """
    Factory function to get a PyTorch optimizer from from an OptimizerConfig instance.

    Parameters
    ----------
    config : OptimizerConfig
        the configuration object.
    network : nn.Module
        the neural network to optimize.

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
    config = deepcopy(config)
    optimizer_class = getattr(optim, config.name)
    freeze = [] if config.freeze is None else config.freeze

    to_freeze, _ = get_params_in_groups(network, groups=freeze)
    for param in to_freeze:
        param.requires_grad = False

    update_config_with_defaults(config, function=optimizer_class.__init__)
    config_dict = config.model_dump(exclude={"name", "freeze"})

    # deal with parameter groups
    args_by_group, args_global = _regroup_args_by_param_group(config_dict)
    if len(args_by_group) == 0:  # no parameter groups
        params = network.parameters()
    else:
        params = []
        args_by_group = sorted(
            args_by_group.items()
        )  # order in the list is important to match lr_scheduler
        for group, args in args_by_group:
            params_in_group, _ = get_params_in_groups(network, group)
            args.update({"params": params_in_group})
            params.append(args)

        other_params, other_param_names = get_params_not_in_groups(
            network, groups=[group for group, _ in args_by_group]
        )
        if len(other_param_names) > 0:
            params.append({"params": other_params})

    optimizer = optimizer_class(params, **args_global)

    return optimizer, config


def _regroup_args_by_param_group(
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
