from pydantic import computed_field

from .base import MetricConfig

__all__ = [
    "MSEConfig",
    "MAEConfig",
    "RMSEConfig",
]


# TODO : R2 missing
class MSEConfig(MetricConfig):
    "Config class for MSE."

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "MSEMetric"


class MAEConfig(MetricConfig):
    "Config class for MAE."

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "MAEMetric"


class RMSEConfig(MetricConfig):
    "Config class for RMSE."

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "RMSEMetric"
