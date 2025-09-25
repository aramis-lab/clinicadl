from typing import Any, Union

# pylint: disable=unused-import
from .base import MetricConfig
from .classification import (
    AveragePrecisionMetricConfig,
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from .enum import ImplementedMetric
from .loss import LossMetricConfig
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


def get_metric_config(
    name: Union[str, ImplementedMetric],
    **kwargs: Any,
) -> MetricConfig:
    """
    Factory function to get a  metric configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedMetric]
        the name of the metric. Check our documentation to know available metrics.
    **kwargs : Any
        any parameter of the metric. Check our documentation on metrics to
        know these parameters.

    Returns
    -------
    MetricConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    metric = ImplementedMetric(name).value
    config_name = f"{metric}Config"
    config = globals()[config_name]

    return config(**kwargs)
