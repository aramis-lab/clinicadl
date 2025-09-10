import monai
import monai.metrics

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import get_defaults_from

from ..enum import Optimum
from .base import (
    MetricConfig,
    _GetNotNansConfig,
)
from .enum import Average, ConfusionMatrixMetricName

__all__ = [
    "ROCAUCMetricConfig",
    "ConfusionMatrixMetricConfig",
    "AveragePrecisionMetricConfig",
]

ROC_AUC_METRIC_MONAI_DEFAULTS = get_defaults_from(monai.metrics.rocauc.ROCAUCMetric)
CONFUSION_METRICS_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.confusion_matrix.ConfusionMatrixMetric
)
AVERAGE_PRECISION_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.AveragePrecisionMetric
)


class ROCAUCMetricConfig(MetricConfig):
    """
    Config class for :py:class:`monai.metrics.ROCAUCMetric`.
    """

    average: Average = ROC_AUC_METRIC_MONAI_DEFAULTS["average"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class ConfusionMatrixMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.ConfusionMatrixMetric`.
    """

    metric_name: ConfusionMatrixMetricName = CONFUSION_METRICS_MONAI_DEFAULTS[
        "metric_name"
    ]
    include_background: bool = CONFUSION_METRICS_MONAI_DEFAULTS["include_background"]
    compute_sample: bool = CONFUSION_METRICS_MONAI_DEFAULTS["compute_sample"]
    reduction: Reduction = CONFUSION_METRICS_MONAI_DEFAULTS["reduction"]

    def optimum(self) -> Optimum:  # pylint: disable=arguments-differ
        """The optimum of the metric."""
        if self.metric_name in [
            ConfusionMatrixMetricName.MISS_RATE.value,
            ConfusionMatrixMetricName.FALSE_NEGATIVE_RATE.value,
            ConfusionMatrixMetricName.FNR.value,
            ConfusionMatrixMetricName.FALL_OUT.value,
            ConfusionMatrixMetricName.FALSE_POSITIVE_RATE.value,
            ConfusionMatrixMetricName.FPR.value,
            ConfusionMatrixMetricName.FALSE_DISCOVERY_RATE.value,
            ConfusionMatrixMetricName.FDR.value,
            ConfusionMatrixMetricName.FALSE_OMISSION_RATE.value,
            ConfusionMatrixMetricName.FOR.value,
            ConfusionMatrixMetricName.PREVALENCE_THRESHOLD.value,
            ConfusionMatrixMetricName.PT.value,
        ]:
            return Optimum.MIN
        return Optimum.MAX


class AveragePrecisionMetricConfig(MetricConfig):
    """
    Config class for :py:class:`monai.metrics.AveragePrecisionMetric`.
    """

    average: Average = AVERAGE_PRECISION_MONAI_DEFAULTS["average"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
