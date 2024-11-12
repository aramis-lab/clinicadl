from pydantic import computed_field

from .base import MetricConfig, _MetricWithNotNansConfig, _MetricWithReductionConfig
from .enum import ImplementedMetric

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]


# TODO : R2 missing
class MSEMetricConfig(
    MetricConfig, _MetricWithReductionConfig, _MetricWithNotNansConfig
):
    "Config class for MSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MSE


class MAEMetricConfig(
    MetricConfig, _MetricWithReductionConfig, _MetricWithNotNansConfig
):
    "Config class for MAE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.MAE


class RMSEMetricConfig(
    MetricConfig, _MetricWithReductionConfig, _MetricWithNotNansConfig
):
    "Config class for RMSE."

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.RMSE
