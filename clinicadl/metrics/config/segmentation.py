from typing import Optional, Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfig,
    _GetNotNansConfig,
    _IncludeBackgroundConfig,
    _ReductionConfig,
)
from .enum import (
    DistanceMetric,
    GeneralizedDiceScoreReduction,
    ImplementedMetric,
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

    ignore_empty: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES


class DiceMetricConfig(MetricConfig, _BaseSegmentationMetricConfig):
    """Config class for Dice score."""

    num_classes: Union[
        Optional[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    return_with_label: bool = False

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.DICE

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX

    @field_validator("return_with_label", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        assert (
            not v
        ), "return_with_label not supported in ClinicaDL. Please set to False."

        return v


class MeanIoUConfig(MetricConfig, _BaseSegmentationMetricConfig):
    """Config class for IoU metric."""

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.IOU

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class GeneralizedDiceScoreConfig(MetricConfig, _IncludeBackgroundConfig):
    """Config class for generalized Dice score."""

    reduction: Union[
        GeneralizedDiceScoreReduction, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    weight_type: Union[WeightType, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.GENERALIZED_DICE

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class _BaseSurfaceDistanceConfig(
    _IncludeBackgroundConfig, _GetNotNansConfig, _ReductionConfig
):
    """Base config class for surface-distance-based metrics."""

    distance_metric: Union[DistanceMetric, DefaultFromLibrary] = DefaultFromLibrary.YES


class SurfaceDistanceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """Config class for Surface Distance metric."""

    symmetric: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.SURF_DIST

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


class HausdorffDistanceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """Config class for Hausdorff distance."""

    percentile: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    directed: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.HAUSDORFF

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN

    @field_validator("percentile", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 100
            ), f"percentile must be between 0 and 100. You passed: {v}."

        return v


class SurfaceDiceMetricConfig(MetricConfig, _BaseSurfaceDistanceConfig):
    """Config class for (normalized) surface Dice score."""

    class_thresholds: Tuple[NonNegativeFloat, ...]
    use_subvoxels: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.SURF_DICE

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX
