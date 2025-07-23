from typing import Optional, Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import get_defaults_from

from .base import Bounds, MaskingMethodConfig, TorchioTransformConfig
from .enum import AnatomicalLabel

__all__ = [
    "RescaleIntensityConfig",
    "ZNormalizationConfig",
    "MaskConfig",
    "ClampConfig",
]

RESCALE_INTENSITY_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RescaleIntensity)
Z_NORMALIZATION_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.ZNormalization)
MASK_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.Mask)
CLAMP_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.Clamp)


class RescaleIntensityConfig(TorchioTransformConfig, MaskingMethodConfig):
    """
    Config class for :py:class:`torchio.transforms.RescaleIntensity`.
    """

    out_min_max: Union[
        NonNegativeFloat, Tuple[float, float]
    ] = RESCALE_INTENSITY_TORCHIO_DEFAULTS["out_min_max"]
    percentiles: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]
    ] = RESCALE_INTENSITY_TORCHIO_DEFAULTS["percentiles"]
    masking_method: Optional[
        Union[str, AnatomicalLabel, Bounds]
    ] = RESCALE_INTENSITY_TORCHIO_DEFAULTS["masking_method"]
    in_min_max: Optional[
        Union[NonNegativeFloat, Tuple[float, float]]
    ] = RESCALE_INTENSITY_TORCHIO_DEFAULTS["in_min_max"]

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


class ZNormalizationConfig(TorchioTransformConfig, MaskingMethodConfig):
    """
    Config class for :py:class:`torchio.transforms.ZNormalization`.
    """

    masking_method: Optional[
        Union[str, AnatomicalLabel, Bounds]
    ] = Z_NORMALIZATION_TORCHIO_DEFAULTS["masking_method"]


class MaskConfig(TorchioTransformConfig, MaskingMethodConfig):
    """
    Config class for :py:class:`torchio.transforms.Mask`.
    """

    outside_value: float = MASK_TORCHIO_DEFAULTS["outside_value"]
    labels: Optional[Tuple[int, ...]] = MASK_TORCHIO_DEFAULTS["labels"]


class ClampConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.Clamp`.
    """

    out_min: Optional[float] = CLAMP_TORCHIO_DEFAULTS["out_min"]
    out_max: Optional[float] = CLAMP_TORCHIO_DEFAULTS["out_max"]

    @model_validator(mode="after")
    def validate_min_max(self):
        """Checks consistency between 'out_min' and 'out_max'."""
        if self.out_min is None and self.out_max is None:
            raise ValueError("'out_min' and 'out_max' cannot both be None.")
        elif self.out_min and self.out_max and self.out_min > self.out_max:
            raise ValueError(
                f"'out_min' should be smaller than 'out_max'. Got out_min={self.out_min} and out_max={self.out_max}"
            )

        return self
