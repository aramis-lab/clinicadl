from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=TransformConfig,
    enum=ImplementedTransform,
    context=globals(),
    config=True,
)
def get_transform_from_dict(data: dict[str, Any]) -> TransformConfig:
    """
    Factory function to get a :py:class:`TransformConfig` from the
    dictionary returned by :py:meth:`TransformConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    TransformConfig
        The config class, parametrized with the input dictionary.
    """
