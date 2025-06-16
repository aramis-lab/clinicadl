from typing import Union

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig, _GetNotNansConfig, _ReductionConfig
from .enum import Optimum

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
        """
        Config class for the Mean Squared Error (MSE) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.MSEMetric
        """
        super().__init__(
            reduction=reduction,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class MAEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MAEMetric`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Mean Absolute Error (MAE) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.MAEMetric
        """
        super().__init__(
            reduction=reduction,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class RMSEMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.RMSEMetric`.
    """

    def __init__(
        self,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Root Mean Squared Error (RMSE) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.RMSEMetric
        """

        super().__init__(
            reduction=reduction,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN
