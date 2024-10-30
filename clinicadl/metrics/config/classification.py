from abc import ABC, abstractmethod
from typing import Type, Union

from pydantic import computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig
from .enum import Average, ConfusionMatrixMetric

__all__ = [
    "ROCAUCConfig",
    "create_confusion_matrix_config",
]


# TODO : AP is missing
class ROCAUCConfig(MetricConfig):
    "Config class for ROC AUC."

    average: Union[Average, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "ROCAUCMetric"


class ConfusionMatrixMetricConfig(MetricConfig, ABC):
    "Config class for metrics derived from the confusion matrix."

    compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "ConfusionMatrixMetric"

    @computed_field
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """The name of the metric computed from the confusion matrix."""


def create_confusion_matrix_config(
    metric_name: ConfusionMatrixMetric,
) -> Type[ConfusionMatrixMetricConfig]:
    """
    Builds a config class for a specific metric computed from the confusion matrix."

    Parameters
    ----------
    metric_name : ConfusionMatrixMetric
        The metric name (e.g. 'f1 score', 'accuracy', etc.).

    Returns
    -------
    Type[ConfusionMatrixMetricConfig]
        The config class.
    """

    class ConfusionMatrixMetricSubConfig(ConfusionMatrixMetricConfig):
        "A sub config class for a specific metric computed from the confusion matrix."

        @computed_field
        @property
        def metric_name(self) -> str:
            """The name of the metric computed from the confusion matrix."""
            return metric_name

    return ConfusionMatrixMetricSubConfig
