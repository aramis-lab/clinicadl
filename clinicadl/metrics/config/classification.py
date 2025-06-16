from typing import Union

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import DefaultFromLibrary

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


# TODO : AP is missing
class ROCAUCMetricConfig(MetricConfig):
    """
    Config class for :py:class:`monai.metrics.ROCAUCMetric`.
    """

    average: Average

    def __init__(
        self, average: Union[Average, DefaultFromLibrary] = DefaultFromLibrary.YES
    ):
        """
        Config class for the ROC AUC metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ROCAUCMetric
        """
        super().__init__(average=average)  # type: ignore

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

    metric_name: Union[ConfusionMatrixMetricName, list[ConfusionMatrixMetricName]]
    compute_sample: bool

    def __init__(
        self,
        metric_name: Union[
            Union[str, ConfusionMatrixMetricName],
            list[Union[str, ConfusionMatrixMetricName]],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Confusion Matrix metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
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


# FOLLOWING METRICS ARE NOT MEANT TO BE USED FOR NOW


class SensitivityMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Sensitivity metric."""

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Sensitivity metric. \n
        (Sensitivity is also known as True Positive Rate (TPR), Hit Rate or Recall.) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        self.metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.SE

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class SpecificityMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Specificity metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.SP

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Specificity metric. \n
        (Specificity is also known as Selectivity or True Negative Rate (TNR).) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class PrecisionMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Precision metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.P

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Precision metric. \n
        (Precision is also known as Positive Predictive Value (PPV).) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class NegativePredictiveValueMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for NegativePredictiveValue metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.NPV

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Negative Predictive Value (NPV) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class MissRateMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for MissRate metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.MISS_RATE

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Miss Rate metric. \n
        (Miss Rate is also known as False Negative Rate (FNR).) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class FallOutMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for FallOut metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.FALL_OUT

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Fall Out metric. \n
        (Fall Out is also known as False Positive Rate (FPR).) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class FalseDiscoveryRateMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for FalseDiscoveryRate metric."""

    metric_name: ConfusionMatrixMetricName = (
        ConfusionMatrixMetricName.FALSE_DISCOVERY_RATE
    )

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the False Discovery Rate (FDR) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class FalseOmissionRateMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for FalseOmissionRate metric."""

    metric_name: ConfusionMatrixMetricName = (
        ConfusionMatrixMetricName.FALSE_OMISSION_RATE
    )

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the False Omission Rate (FOR) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class PrevalenceThresholdMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for PrevalenceThreshold metric."""

    metric_name: ConfusionMatrixMetricName = (
        ConfusionMatrixMetricName.PREVALENCE_THRESHOLD
    )

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Prevalence Threshold (PT) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class ThreatScoreMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for ThreatScore metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.THREAT_SCORE

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Threat Score (TS) metric. \n
        (Threat Score is also known as Critical Success Index (CSI)) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """

        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class AccuracyMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Accuracy metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.ACCURACY

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Accuracy metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class BalancedAccuracyMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for BalancedAccuracy metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.BALANCED_ACCURACY

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Balanced Accuracy (BA) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class F1ScoreMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for F1Score metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.F1_SCORE

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the F1 Score metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class MatthewsCoeffMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for MatthewsCoeff metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.MATTHEWS_COEFF

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Matthews Correlation Coefficient (MCC) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class FowlkesMallowsMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for FowlkesMallows metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.FOWLKES_MALLOWS

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Fowlkes-Mallows Index (FM) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class InformednessMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Informedness metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.INFORMEDNESS

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Informedness metric. \n
        (Informedness is also known as Bookmaker Informedness (BM) or Youden Index.) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class MarkednessMetricConfig(ConfusionMatrixMetricConfig):
    """Config class for Markedness metric."""

    metric_name: ConfusionMatrixMetricName = ConfusionMatrixMetricName.MARKEDNESS

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        compute_sample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Markedness (MK) metric. \n
        (Markedness is also known as Deltap metric.) \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.ConfusionMatrixMetric
        """
        super().__init__(
            include_background=include_background,
            metric_name=self.metric_name,
            compute_sample=compute_sample,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @classmethod
    def optimum(cls) -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
