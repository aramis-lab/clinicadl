from typing import Union

import monai
import monai.metrics

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import get_defaults_from

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import Average, ConfusionMatrixMetricName, Optimum

__all__ = [
    "ROCAUCMetricConfig",
    "ConfusionMatrixMetricConfig",
]

ROC_AUC_METRIC_METRICS_DEFAULTS = get_defaults_from(monai.metrics.rocauc.ROCAUCMetric)
CONFUSION_METRICS_DEFAULTS = get_defaults_from(
    monai.metrics.confusion_matrix.ConfusionMatrixMetric
)


# TODO : AP is missing
class ROCAUCMetricConfig(MetricConfig):
    """
    Config class for :py:class:`monai.metrics.ROCAUCMetric`.
    """

    average: Average = ROC_AUC_METRIC_METRICS_DEFAULTS["average"]

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

    metric_name: Union[
        ConfusionMatrixMetricName, list[ConfusionMatrixMetricName]
    ] = CONFUSION_METRICS_DEFAULTS["metric_name"]
    include_background: bool = CONFUSION_METRICS_DEFAULTS["include_background"]
    compute_sample: bool = CONFUSION_METRICS_DEFAULTS["compute_sample"]
    get_not_nans: bool = CONFUSION_METRICS_DEFAULTS["get_not_nans"]
    reduction: Reduction = CONFUSION_METRICS_DEFAULTS["reduction"]

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
