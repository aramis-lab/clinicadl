from typing import Optional, Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    MetricConfigWithBackground,
    MetricConfigWithNotNans,
    MetricConfigWithReduction,
)
from .enum import (
    DistanceMetric,
    GeneralizedDiceScoreReduction,
    ImplementedMetric,
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


class BaseSegmentationMetricConfig(
    MetricConfigWithBackground, MetricConfigWithNotNans, MetricConfigWithReduction
):
    """Base config class for segmentation metrics."""

    ignore_empty: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES


class DiceMetricConfig(BaseSegmentationMetricConfig):
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

    @field_validator("return_with_label", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        assert (
            not v
        ), "return_with_label not supported in ClinicaDL. Please set to False."

        return v


class MeanIoUConfig(BaseSegmentationMetricConfig):
    """Config class for IoU metric."""

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.IOU


class GeneralizedDiceScoreConfig(MetricConfigWithBackground):
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


class BaseSurfaceDistanceConfig(
    MetricConfigWithBackground, MetricConfigWithNotNans, MetricConfigWithReduction
):
    """Base config class for surface-distance-based metrics."""

    distance_metric: Union[DistanceMetric, DefaultFromLibrary] = DefaultFromLibrary.YES


class SurfaceDistanceMetricConfig(BaseSurfaceDistanceConfig):
    """Config class for Surface Distance metric."""

    symmetric: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.SURF_DIST


class HausdorffDistanceMetricConfig(BaseSurfaceDistanceConfig):
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

    @field_validator("percentile", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 100
            ), f"percentile must be between 0 and 100. You passed: {v}."

        return v


class SurfaceDiceMetricConfig(BaseSurfaceDistanceConfig):
    """Config class for (normalized) surface Dice score."""

    class_thresholds: Tuple[NonNegativeFloat, ...]
    use_subvoxels: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedMetric:
        """The name of the metric."""
        return ImplementedMetric.SURF_DICE
