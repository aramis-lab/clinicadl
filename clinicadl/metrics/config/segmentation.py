from typing import Optional, Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, field_validator

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import (
    DistanceMetric,
    Optimum,
    WeightType,
)

__all__ = [
    "DiceMetricConfig",
    "MeanIoUConfig",
    "GeneralizedDiceScoreConfig",
    "SurfaceDistanceMetricConfig",
    "HausdorffDistanceMetricConfig",
    "SurfaceDiceMetricConfig",
]


class _BaseSegmentationMetricConfig(
    _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    """Base config class for segmentation metrics."""

    ignore_empty: bool


class DiceMetricConfig(MetricConfig, _BaseSegmentationMetricConfig):
    """
    Config class for :py:class:`monai.metrics.DiceMetric`.
    """

    num_classes: Optional[PositiveInt]
    return_with_label: bool = False

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        ignore_empty: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        num_classes: Union[
            Optional[PositiveInt], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        return_with_label: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Dice metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.DiceMetric
        """
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty,
            num_classes=num_classes,
            return_with_label=return_with_label,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX

    @field_validator("return_with_label", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        assert (
            not v
        ), "'return_with_label' not supported in ClinicaDL. Please leave to False."

        return v


class MeanIoUConfig(MetricConfig, _BaseSegmentationMetricConfig):
    """
    Config class for :py:class:`monai.metrics.MeanIoU`.
    """

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        ignore_empty: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Mean Intersection over Union (MeanIoU) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.MeanIoU
        """

        super().__init__(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
            ignore_empty=ignore_empty,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class GeneralizedDiceScoreConfig(MetricConfig, _IncludeBackgroundConfig):
    """
    Config class for :py:class:`monai.metrics.GeneralizedDiceScore`.
    """

    reduction: Reduction
    weight_type: WeightType

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Reduction = Reduction.MEAN,  # TODO: check how to deal with mean_batch before MONAI 1.5
        weight_type: Union[WeightType, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Generalized Dice Score metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.GeneralizedDiceScore
        """
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            weight_type=weight_type,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class _BaseSurfaceDistanceConfig(
    _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    """Base config class for surface-distance-based metrics."""

    distance_metric: DistanceMetric


class SurfaceDistanceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """
    Config class for :py:class:`monai.metrics.SurfaceDistanceMetric`.
    """

    symmetric: bool

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        symmetric: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        distance_metric: Union[
            DistanceMetric, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = (DefaultFromLibrary.YES),
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Surface Distance metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.SurfaceDistanceMetric
        """
        super().__init__(
            include_background=include_background,
            symmetric=symmetric,
            distance_metric=distance_metric,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class HausdorffDistanceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """
    Config class for :py:class:`monai.metrics.HausdorffDistanceMetric`.
    """

    percentile: Optional[NonNegativeFloat]
    directed: bool

    def __init__(
        self,
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        distance_metric: Union[
            DistanceMetric, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        percentile: Union[Optional[NonNegativeFloat], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        directed: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = (DefaultFromLibrary.YES),
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Hausdorff Distance metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.HausdorffDistanceMetric
        """
        super().__init__(
            include_background=include_background,
            distance_metric=distance_metric,
            percentile=percentile,
            directed=directed,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN

    @field_validator("percentile", mode="after")
    @classmethod
    def validator_percentile(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 100
            ), f"percentile must be between 0 and 100. You passed: {v}."

        return v


class SurfaceDiceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """
    Config class for :py:class:`monai.metrics.SurfaceDiceMetric`.
    """

    class_thresholds: Tuple[NonNegativeFloat, ...]
    use_subvoxels: bool

    def __init__(
        self,
        class_thresholds: Tuple[NonNegativeFloat, ...],
        include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        distance_metric: Union[
            DistanceMetric, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        use_subvoxels: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Surface Dice metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.SurfaceDiceMetric
        """

        super().__init__(
            class_thresholds=class_thresholds,
            include_background=include_background,
            distance_metric=distance_metric,
            reduction=reduction,
            get_not_nans=get_not_nans,
            use_subvoxels=use_subvoxels,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
