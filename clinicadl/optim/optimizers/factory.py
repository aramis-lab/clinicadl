from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=OptimizerConfig,
    enum=ImplementedOptimizer,
    context=globals(),
    config=True,
)
def get_optimizer_from_dict(data: dict[str, Any]) -> OptimizerConfig:
    """
    Factory function to get a :py:class:`OptimizerConfig` from the
    dictionary returned by :py:meth:`OptimizerConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    OptimizerConfig
        The config class, parametrized with the input dictionary.
    """
