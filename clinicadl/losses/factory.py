from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=LossConfig, enum=ImplementedLoss, context=globals(), config=True
)
def get_loss_function_from_dict(data: dict[str, Any]) -> LossConfig:
    """
    Factory function to get a :py:class:`LossConfig` from the
    dictionary returned by :py:meth:`LossConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    LossConfig
        The config class, parametrized with the input dictionary.
    """
