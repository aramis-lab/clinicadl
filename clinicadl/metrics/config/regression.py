import monai

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import get_defaults_from

from ..enum import Optimum
from .base import MetricConfig, _GetNotNansConfig

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]

MSE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.MSEMetric)
MAE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.MAEMetric)
RMSE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.RMSEMetric)


# TODO : R2 missing
class MSEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MSEMetric`.
    """

    reduction: Reduction = MSE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class MAEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MAEMetric`.
    """

    reduction: Reduction = MAE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class RMSEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.RMSEMetric`.
    """

    reduction: Reduction = RMSE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN
