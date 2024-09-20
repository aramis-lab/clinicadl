from copy import deepcopy
from typing import Tuple

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import LRSchedulerConfig


def get_lr_scheduler(
    optimizer: optim.Optimizer, config: LRSchedulerConfig
) -> Tuple[lr_scheduler.LRScheduler, LRSchedulerConfig]:
    """
    Factory function to get a LR scheduler from PyTorch.

    Parameters
    ----------
    optimizer : optim.Optimizer
        The optimizer to schedule.
    config : LRSchedulerConfig
        The config class with the parameters of the LR scheduler.

    Returns
    -------
    lr_scheduler.LRScheduler
        The LR scheduler.
    LRSchedulerConfig
        The updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    if config.scheduler is None:
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1), config

    scheduler_class = getattr(lr_scheduler, config.scheduler)
    expected_args, config_dict = get_args_and_defaults(scheduler_class.__init__)
    for arg, value in config.model_dump().items():
        if arg in expected_args and value != DefaultFromLibrary.YES:
            config_dict[arg] = value

    config_dict_ = deepcopy(config_dict)
    if "min_lr" in config_dict and isinstance(config_dict["min_lr"], dict):
        config_dict_["min_lr"] = [
            v for group, v in sorted(config_dict["min_lr"].items()) if group != "ELSE"
        ]  # order in the list is important
        if "ELSE" in config_dict["min_lr"]:
            config_dict_["min_lr"].append(
                config_dict["min_lr"]["ELSE"]
            )  # ELSE must be the last group
    scheduler = scheduler_class(optimizer, **config_dict_)

    updated_config = config.model_copy(update=config_dict)

    return scheduler, updated_config
