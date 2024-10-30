from typing import Optional, Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig
from .enum import DistanceMetric, GeneralizedDiceScoreReduction, WeightType

__all__ = [
    "DiceConfig",
    "IoUConfig",
    "GeneralizedDiceConfig",
    "SurfaceDistanceConfig",
    "HausdorffDistanceConfig",
    "SurfaceDiceConfig",
]


class SegmentationMetricConfig(MetricConfig):
    """Base config class for segmentation metrics."""

    ignore_empty: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES


class DiceConfig(SegmentationMetricConfig):
    """Config class for Dice score."""

    num_classes: Union[
        Optional[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    return_with_label: bool = False

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "DiceMetric"

    @field_validator("return_with_label", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        assert (
            not v
        ), "return_with_label not supported in ClinicaDL. Please set to False."

        return v


class IoUConfig(SegmentationMetricConfig):
    """Config class for IoU metric."""

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "MeanIoU"


class GeneralizedDiceConfig(MetricConfig):
    """Config class for generalized Dice score."""

    reduction: Union[
        GeneralizedDiceScoreReduction, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    weight_type: Union[WeightType, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "GeneralizedDiceScore"


class SurfaceDistanceConfig(MetricConfig):
    """Config class for Surface Distance metric."""

    distance_metric: Union[DistanceMetric, DefaultFromLibrary] = DefaultFromLibrary.YES
    symmetric: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "SurfaceDistanceMetric"


class HausdorffDistanceConfig(SurfaceDistanceConfig):
    """Config class for Hausdorff distance."""

    percentile: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    directed: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "HausdorffDistanceMetric"

    @field_validator("percentile", mode="after")
    @classmethod
    def validator_return_with_label(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 100
            ), f"percentile must be between 0 and 100. You passed: {v}."

        return v


class SurfaceDiceConfig(SurfaceDistanceConfig):
    """Config class for (normalized) surface Dice score."""

    class_thresholds: Tuple[NonNegativeFloat, ...]
    use_subvoxels: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "SurfaceDiceMetric"
