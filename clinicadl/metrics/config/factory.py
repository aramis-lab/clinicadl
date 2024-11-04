from typing import Type, Union

# pylint: disable=unused-import
from .base import MetricConfig
from .classification import ConfusionMatrixMetricConfig, ROCAUCMetricConfig
from .enum import ImplementedMetric
from .reconstruction import (
    MultiScaleSSIMMetricConfig,
    PSNRMetricConfig,
    SSIMMetricConfig,
)
from .regression import MAEMetricConfig, MSEMetricConfig, RMSEMetricConfig
from .segmentation import (
    DiceMetricConfig,
    GeneralizedDiceScoreConfig,
    HausdorffDistanceMetricConfig,
    MeanIoUConfig,
    SurfaceDiceMetricConfig,
    SurfaceDistanceMetricConfig,
)


def create_metric_config(
    metric: Union[str, ImplementedMetric],
) -> Type[MetricConfig]:
    """
    A factory function to create a config class suited for the metric.

    Parameters
    ----------
    metric : Union[str, ImplementedMetric]
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
    metric = ImplementedMetric(metric)
    if metric == ImplementedMetric.LOSS:
        raise ValueError(
            "To use the loss as a metric, please use directly clinicadl.metrics.loss_to_metric."
        )

    config_name = "".join([metric, "Config"])
    config = globals()[config_name]

    return config
