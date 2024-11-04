from typing import Union

from pydantic import computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    MetricConfigWithBackground,
    MetricConfigWithNotNans,
    MetricConfigWithReduction,
)
from .enum import Average, ImplementedMetric

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


class ConfusionMatrixMetricConfig(
    MetricConfigWithBackground, MetricConfigWithNotNans, MetricConfigWithReduction
):
    "Config class for metrics derived from the confusion matrix."

    metric_name: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES
    compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.CONF_MATRIX
