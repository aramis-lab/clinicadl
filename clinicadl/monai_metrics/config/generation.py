from pydantic import PositiveFloat, computed_field

from .base import MetricConfig

__all__ = ["MMDMetricConfig"]


class MMDMetricConfig(MetricConfig):
    "Config class for MMD metric."

    kernel_bandwidth: PositiveFloat = 1.0

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "MMDMetric"
