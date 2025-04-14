from typing import Union

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import Average, ConfusionMatrixMetricName, Optimum, Reduction

__all__ = [
    "ROCAUCMetricConfig",
    "ConfusionMatrixMetricConfig",
]


# TODO : AP is missing
class ROCAUCMetricConfig(MetricConfig):
    """
    Config class for :py:class:`monai.metrics.ROCAUCMetric`.
    """

    average: Average

    def __init__(
        self, average: Union[Average, DefaultFromLibrary] = DefaultFromLibrary.YES
    ):
        super().__init__(average=average)

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class ConfusionMatrixMetricConfig(
    MetricConfig, _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    """
    Config class for :py:class:`monai.metrics.ConfusionMatrixMetric`.
    """

    metric_name: ConfusionMatrixMetricName
    compute_sample: bool

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        metric_name: Union[
            ConfusionMatrixMetricName, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            include_background=include_background,
            metric_name=metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    def optimum(self) -> Optimum:  # pylint: disable=arguments-differ
        """The optimum of the metric."""
        if self.metric_name in [
            "miss_rate",
            "false_negative_rate",
            "fnr",
            "fall_out",
            "false_positive_rate",
            "fpr",
            "false_discovery_rate",
            "fdr",
            "false_omission_rate",
            "for",
            "prevalence_threshold",
            "pt",
        ]:
            return Optimum.MIN
        return Optimum.MAX
