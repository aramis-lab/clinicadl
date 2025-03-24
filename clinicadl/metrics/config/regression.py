from typing import Union

import monai.metrics as metrics
from pydantic import computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig, _GetNotNansConfig, _ReductionConfig
from .enum import ImplementedMetric, Reduction

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]


# TODO : R2 missing
class MSEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MSEMetric`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return ImplementedMetric.MSE.value

    def _get_class(self) -> type[metrics.Metric]:
        """Returns the metric associated to this config class."""
        return metrics.MSEMetric


class MAEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MAEMetric`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return ImplementedMetric.MAE.value

    def _get_class(self) -> type[metrics.Metric]:
        """Returns the metric associated to this config class."""
        return metrics.MAEMetric


class RMSEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.RMSEMetric`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            reduction=reduction,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return ImplementedMetric.RMSE.value

    def _get_class(self) -> type[metrics.Metric]:
        """Returns the metric associated to this config class."""
        return metrics.RMSEMetric
