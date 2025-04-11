from typing import Union

from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig, _GetNotNansConfig, _ReductionConfig
from .enum import Reduction

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
