from typing import Any, Union

import clinicadl.optim.lr_schedulers.config as lr_schedulers

from .configs import LRSchedulerConfig
from .enum import ImplementedLRScheduler


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
        The name of the lr scheduler. Check our documentation to know
        available schedulers.
    **kwargs : Any
        Any parameter of the lr scheduler. Check our documentation on lr schedulers to
        know these parameters.

    Returns
    -------
    LRSchedulerConfig
        The configuration object. Default values will be returned for the parameters
        not passed by the user.
    """
    lr_scheduler = lr_schedulers.ImplementedLRScheduler(name)
    config_name = f"{lr_scheduler.value}Config"
    config = getattr(lr_schedulers, config_name)

    return config(**kwargs)
