from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=NetworkConfig, enum=ImplementedNetwork, context=globals(), config=True
)
def get_network_from_dict(data: dict[str, Any]) -> NetworkConfig:
    """
    Factory function to get a :py:class:`NetworkConfig` from the
    dictionary returned by :py:meth:`NetworkConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    NetworkConfig
        The config class, parametrized with the input dictionary.
    """
