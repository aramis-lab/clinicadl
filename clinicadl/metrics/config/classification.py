from typing import Union

import monai.metrics as metrics
from pydantic import computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import Average, ConfusionMatrixMetricName, ImplementedMetric, Reduction

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

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return ImplementedMetric.ROC_AUC.value

    def _get_class(self) -> type[metrics.Metric]:
        """Returns the metric associated to this config class."""
        return metrics.ROCAUCMetric


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

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return ImplementedMetric.CONF_MATRIX.value

    def _get_class(self) -> type[metrics.Metric]:
        """Returns the metric associated to this config class."""
        return metrics.ConfusionMatrixMetric
