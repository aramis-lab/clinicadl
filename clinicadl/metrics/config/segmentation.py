from typing import Optional, Tuple, Union

import monai
import monai.metrics
from pydantic import NonNegativeFloat, PositiveInt, field_validator

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import get_defaults_from

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

DICE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.DiceMetric)
MEAN_IOU_MONAI_DEFAULTS = get_defaults_from(monai.metrics.MeanIoU)
GENERALIZED_DICE_SCORE_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.GeneralizedDiceScore
)
SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.SurfaceDistanceMetric
)
HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.HausdorffDistanceMetric
)
SURFACE_DICE_METRIC_MONAI_DEFAULTS = get_defaults_from(monai.metrics.SurfaceDiceMetric)


class _BaseSegmentationMetricConfig(
    _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    """Base config class for segmentation metrics."""

    ignore_empty: bool


class DiceMetricConfig(MetricConfig, _BaseSegmentationMetricConfig):
    """
    Config class for :py:class:`monai.metrics.DiceMetric`.
    """

    num_classes: Optional[PositiveInt] = DICE_MONAI_DEFAULTS["num_classes"]
    return_with_label: bool = DICE_MONAI_DEFAULTS["return_with_label"]
    include_background: bool = DICE_MONAI_DEFAULTS["include_background"]
    reduction: Reduction = DICE_MONAI_DEFAULTS["reduction"]
    get_not_nans: bool = DICE_MONAI_DEFAULTS["get_not_nans"]
    ignore_empty: bool = DICE_MONAI_DEFAULTS["ignore_empty"]

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

    include_background: bool = MEAN_IOU_MONAI_DEFAULTS["include_background"]
    reduction: Reduction = MEAN_IOU_MONAI_DEFAULTS["reduction"]
    get_not_nans: bool = MEAN_IOU_MONAI_DEFAULTS["get_not_nans"]
    ignore_empty: bool = MEAN_IOU_MONAI_DEFAULTS["ignore_empty"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class GeneralizedDiceScoreConfig(MetricConfig, _IncludeBackgroundConfig):
    """
    Config class for :py:class:`monai.metrics.GeneralizedDiceScore`.
    """

    reduction: Reduction = Reduction.MEAN
    weight_type: WeightType = GENERALIZED_DICE_SCORE_MONAI_DEFAULTS["weight_type"]
    include_background: bool = GENERALIZED_DICE_SCORE_MONAI_DEFAULTS[
        "include_background"
    ]

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

    symmetric: bool = SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS["symmetric"]
    include_background: bool = SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS[
        "include_background"
    ]
    distance_metric: DistanceMetric = SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS[
        "distance_metric"
    ]
    reduction: Reduction = SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS["reduction"]
    get_not_nans: bool = SURFACE_DISTANCE_METRIC_MONAI_DEFAULTS["get_not_nans"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class HausdorffDistanceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """
    Config class for :py:class:`monai.metrics.HausdorffDistanceMetric`.
    """

    percentile: Optional[NonNegativeFloat] = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS[
        "percentile"
    ]
    directed: bool = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS["directed"]
    include_background: bool = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS[
        "include_background"
    ]
    distance_metric: DistanceMetric = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS[
        "distance_metric"
    ]
    reduction: Reduction = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS["reduction"]
    get_not_nans: bool = HAUSDORFF_DISTANCE_METRIC_MONAI_DEFAULTS["get_not_nans"]

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
    use_subvoxels: bool = SURFACE_DICE_METRIC_MONAI_DEFAULTS["use_subvoxels"]
    include_background: bool = SURFACE_DICE_METRIC_MONAI_DEFAULTS["include_background"]
    distance_metric: DistanceMetric = SURFACE_DICE_METRIC_MONAI_DEFAULTS[
        "distance_metric"
    ]
    reduction: Reduction = SURFACE_DICE_METRIC_MONAI_DEFAULTS["reduction"]
    get_not_nans: bool = SURFACE_DICE_METRIC_MONAI_DEFAULTS["get_not_nans"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
