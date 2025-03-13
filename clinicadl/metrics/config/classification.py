from typing import Union

from pydantic import computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import Average, ImplementedMetric, Optimum

__all__ = [
    "ROCAUCMetricConfig",
    "ConfusionMatrixMetricConfig",
]


# TODO : AP is missing
class ROCAUCMetricConfig(MetricConfig):
    "Config class for ROC AUC."

    average: Union[Average, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.ROC_AUC

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class ConfusionMatrixMetricConfig(
    MetricConfig, _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    "Config class for metrics derived from the confusion matrix."

    metric_name: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES
    compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.CONF_MATRIX

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
