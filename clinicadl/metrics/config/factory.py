from typing import Type, Union

from .base import MetricConfig
from .classification import *
from .enum import ConfusionMatrixMetric, ImplementedMetrics
from .generation import *
from .reconstruction import *
from .regression import *
from .segmentation import *


def create_metric_config(
    metric: Union[str, ImplementedMetrics],
) -> Type[MetricConfig]:
    """
    A factory function to create a config class suited for the metric.

    Parameters
    ----------
    metric : Union[str, ImplementedMetrics]
        The name of the metric.

    Returns
    -------
    Type[MetricConfig]
        The config class.

    Raises
    ------
    ValueError
        When `metric`does not correspond to any supported metric.
    ValueError
        When `metric` is `Loss`.
    """
    metric = ImplementedMetrics(metric)
    if metric == ImplementedMetrics.LOSS:
        raise ValueError(
            "To use the loss as a metric, please use directly clinicadl.metrics.loss_to_metric."
        )

    # special cases
    if metric == ImplementedMetrics.MS_SSIM:
        return MultiScaleSSIMConfig
    if metric == ImplementedMetrics.MMD:
        return MMDMetricConfig

    try:
        metric = ConfusionMatrixMetric(metric.lower())
        return create_confusion_matrix_config(metric)
    except ValueError:
        pass

    # "normal" cases:
    try:
        config = _get_config(metric)
    except KeyError:
        config = _get_config(metric.title().replace(" ", ""))

    return config


def _get_config(name: str) -> Type[MetricConfig]:
    """
    Tries to get a config class associated to the name.

    Parameters
    ----------
    name : str
        The name of the metric.

    Returns
    -------
    Type[MetricConfig]
        The config class.
    """
    config_name = "".join([name, "Config"])
    return globals()[config_name]
