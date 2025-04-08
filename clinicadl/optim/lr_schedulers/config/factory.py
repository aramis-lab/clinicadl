from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import torch.optim as optim
import torch.optim.lr_scheduler as lr_schedulers

from clinicadl.utils.factories import update_config_with_defaults

from .configs import (
    ImplementedLRScheduler,
    LRSchedulerConfig,
    create_lr_scheduler_config,
)


def get_lr_scheduler_config(
    name: Union[str, ImplementedLRScheduler],
    **kwargs: Any,
) -> LRSchedulerConfig:
    """
    Factory function to get a lr scheduler configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedLRScheduler]
        the name of the lr scheduler. Check our documentation to know
        available schedulers.
    **kwargs : Any
        any parameter of the lr scheduler. Check our documentation on lr schedulers to
        know these parameters.

    Returns
    -------
    LRSchedulerConfig
        the configuration object. Default values will be returned for the parameters
        not passed by the user.
    """
    config = create_lr_scheduler_config(name)(**kwargs)
    scheduler_class = getattr(lr_schedulers, config.name)

    update_config_with_defaults(config, function=scheduler_class.__init__)

    return config


def get_lr_scheduler_from_config(
    config: Optional[LRSchedulerConfig], optimizer: optim.Optimizer
) -> Tuple[lr_schedulers.LRScheduler, LRSchedulerConfig]:
    """
    Factory function to get a LR scheduler from PyTorch.

    Parameters
    ----------
    config : Optional[LRSchedulerConfig]
        the config class with the parameters of the LR scheduler.
        If None, no lr scheduler will be used.
    optimizer : optim.Optimizer
        the optimizer to schedule.

    Returns
    -------
    lr_scheduler.LRScheduler
        the LR scheduler.
    LRSchedulerConfig
        the updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    if config is None:
        return lr_schedulers.LambdaLR(optimizer, lr_lambda=lambda epoch: 1), config

    config = deepcopy(config)
    scheduler_class = getattr(lr_schedulers, config.name)

    update_config_with_defaults(config, function=scheduler_class.__init__)
    config_dict = config.model_dump(exclude={"name"})

    # deal with parameter groups
    if "min_lr" in config_dict and isinstance(config_dict["min_lr"], dict):
        min_lr_by_group = sorted(
            filter(lambda x: x[0] != "ELSE", config_dict["min_lr"].items())
        )  # order in the list is important
        min_lrs = [value for _, value in min_lr_by_group]
        min_lrs.append(config_dict["min_lr"]["ELSE"])  # ELSE must be the last group
        config_dict["min_lr"] = min_lrs

    scheduler = scheduler_class(optimizer, **config_dict)

    return scheduler, config
