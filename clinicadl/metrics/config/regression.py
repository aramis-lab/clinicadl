from pydantic import computed_field

from .base import MetricConfigWithNotNans, MetricConfigWithReduction
from .enum import ImplementedMetric

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]


# TODO : R2 missing
class MSEMetricConfig(MetricConfigWithReduction, MetricConfigWithNotNans):
    "Config class for MSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MSE


class MAEMetricConfig(MetricConfigWithReduction, MetricConfigWithNotNans):
    "Config class for MAE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MAE


class RMSEMetricConfig(MetricConfigWithReduction, MetricConfigWithNotNans):
    "Config class for RMSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.RMSE
