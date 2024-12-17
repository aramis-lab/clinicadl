from typing import Optional, Tuple, Union

from pydantic import (
    PositiveFloat,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedTransform, TransformConfig, _MaskingMethodConfig

__all__ = ["RescaleIntensityConfig", "ZNormalizationConfig", "ClampConfig"]


class RescaleIntensityConfig(TransformConfig, _MaskingMethodConfig):
    """Config class for RescaleIntensity transform."""

    out_min_max: Union[
        float, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    percentiles: Union[
        PositiveFloat, Tuple[PositiveFloat, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    in_min_max: Union[
        Optional[Union[float, Tuple[float, float]]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RESCALE_INTENSITY

    @field_validator("percentiles", mode="after")
    @classmethod
    def validator_percentiles(cls, v):
        """Checks that percentiles are between 0 and 100."""
        if isinstance(v, float):
            cls._check_percentile(v)
        elif isinstance(v, tuple):
            cls._check_percentile(v[0])
            cls._check_percentile(v[1])
            if v[0] > v[1]:
                raise ValueError(
                    f"In 'percentiles', the first percentile should be smaller than the second one. Got{v}"
                )
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
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.Z_NORMALIZATION


class ClampConfig(TransformConfig):
    """Config class for Clamp transform."""

    out_min: Union[Optional[float], DefaultFromLibrary]
    out_max: Union[Optional[float], DefaultFromLibrary]

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.CLAMP

    @model_validator(mode="after")
    def validate_min_max(self):
        """Checks consistency between 'out_min' and 'out_max'."""
        if (self.out_min is None) and (self.out_max is None):
            raise ValueError("'out_min' and 'out_max' cannot both be None.")
        elif (
            (self.out_min is not None) and (self.out_max is not None)
        ) and self.out_min > self.out_max:
            raise ValueError(
                f"'out_min' should be smaller than 'out_max'. Got out_min={self.out_min} and out_max={self.out_max}"
            )

        return self
