from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=LRSchedulerConfig,
    enum=ImplementedLRScheduler,
    context=globals(),
    config=True,
)
def get_lr_scheduler_from_dict(data: dict[str, Any]) -> LRSchedulerConfig:
    """
    Factory function to get a :py:class:`LRSchedulerConfig` from the
    dictionary returned by :py:meth:`LRSchedulerConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    LRSchedulerConfig
        The config class, parametrized with the input dictionary.
    """
