from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .config import *


@factory_from_dict(
    object_type=MetricConfig, enum=ImplementedMetric, context=globals(), config=True
)
def get_metric_from_dict(data: dict[str, Any]) -> MetricConfig:
    """
    Factory function to get a :py:class:`MetricConfig` from the
    dictionary returned by :py:meth:`MetricConfig.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    MetricConfig
        The config class, parametrized with the input dictionary.
    """
