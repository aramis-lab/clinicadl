from typing import Optional, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import Bounds, ImplementedTransform, TransformConfig, _MaskingMethodConfig
from .enum import AnatomicalLabel, TransformType

__all__ = [
    "RescaleIntensityConfig",
    "ZNormalizationConfig",
    "MaskConfig",
    "ClampConfig",
    "NanRemovalConfig",
]


class RescaleIntensityConfig(TransformConfig, _MaskingMethodConfig):
    """Config class for RescaleIntensity transform."""

    out_min_max: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    percentiles: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    in_min_max: Union[
        Optional[Union[NonNegativeFloat, Tuple[float, float]]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RESCALE_INTENSITY.value

    @field_validator("out_min_max", "percentiles", "in_min_max", mode="after")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.field_name
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field_name)
        return v

    @field_validator("percentiles", mode="after")
    @classmethod
    def validator_percentiles(cls, v):
        """Checks that percentiles are between 0 and 100."""
        if isinstance(v, float):
            cls._check_percentile(v)
        elif isinstance(v, tuple):
            cls._check_percentile(v[0])
            cls._check_percentile(v[1])
        return v

    @staticmethod
    def _check_percentile(percentile: float) -> None:
        """Checks a single percentile."""
        if not (0 <= percentile <= 100):
            raise ValueError(
                f"'percentiles' must contain values between 0 and 100. Got {percentile}"
            )


class ZNormalizationConfig(TransformConfig, _MaskingMethodConfig):
    """Config class for ZNormalization transform."""

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.Z_NORMALIZATION.value


class MaskConfig(TransformConfig):
    """Config class for Mask transform."""

    masking_method: Optional[Union[str, AnatomicalLabel, Bounds]]
    outside_value: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES
    labels: Union[
        Optional[Tuple[int, ...]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.MASK.value

    @field_validator("masking_method", mode="before")
    @classmethod
    def validator_masking_method(cls, v):
        """To handle 'masking_method' different types."""
        return _MaskingMethodConfig.validator_masking_method(v)


class ClampConfig(TransformConfig):
    """Config class for Clamp transform."""

    out_min: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES
    out_max: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.CLAMP.value

    @model_validator(mode="after")
    def validate_min_max(self):
        """Checks consistency between 'out_min' and 'out_max'."""
        if (self.out_min is None or self.out_min == DefaultFromLibrary.YES) and (
            self.out_max is None or self.out_max == DefaultFromLibrary.YES
        ):
            raise ValueError("'out_min' and 'out_max' cannot both be None.")
        elif (
            not (self.out_min is None or self.out_min == DefaultFromLibrary.YES)
            and not (self.out_max is None or self.out_max == DefaultFromLibrary.YES)
        ) and self.out_min > self.out_max:
            raise ValueError(
                f"'out_min' should be smaller than 'out_max'. Got out_min={self.out_min} and out_max={self.out_max}"
            )

        return self


class NanRemovalConfig(TransformConfig):
    """Config class for NanRemoval transform."""

    nan: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES
    posinf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES
    neginf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.NAN_REMOVAL.value

    @property
    def _type(self) -> TransformType:
        """The source where the transform can be found."""
        return TransformType.HOMEMADE
