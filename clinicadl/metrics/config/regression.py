from pydantic import computed_field

from .base import MetricConfig, _GetNotNansConfig, _ReductionConfig
from .enum import ImplementedMetric

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]


# TODO : R2 missing
class MSEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    "Config class for MSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MSE


class MAEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    "Config class for MAE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MAE


class RMSEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    "Config class for RMSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.RMSE
