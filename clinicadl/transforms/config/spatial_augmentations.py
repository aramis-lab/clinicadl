from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import TransformConfig, _AnatomicalAxesConfig
from .enum import (
    AnatomicalAxis,
    CenterMode,
    ImplementedTransform,
    InterpolationMode,
    LockedBordersMode,
    NumericalAxis,
    RandomAffinePaddingMode,
)

__all__ = [
    "RandomFlipConfig",
    "RandomAffineConfig",
    "RandomElasticDeformationConfig",
    "RandomAnisotropyConfig",
]


class RandomFlipConfig(TransformConfig, _AnatomicalAxesConfig):
    """Config class for RandomFlip transform."""

    flip_probability: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_FLIP

    @field_validator("flip_probability", mode="after")
    @classmethod
    def validator_flip_probability(cls, v):
        """Checks that 'flip_probability' is a probability."""
        if isinstance(v, float) and not (0 <= v <= 1):
            raise ValueError("'flip_probability' must be between 0 and 1.")
        return v


SpatialRange = Union[
    NonNegativeFloat,
    tuple[float, float],
    Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    Tuple[float, float, float, float, float, float],
]


class RandomAffineConfig(TransformConfig):
    """Config class for RandomAffine transform."""

    scales: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES
    degrees: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES
    translation: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES
    isotropic: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    center: Union[CenterMode, DefaultFromLibrary] = DefaultFromLibrary.YES
    default_pad_value: Union[
        float, RandomAffinePaddingMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    image_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    label_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    check_shape: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_AFFINE

    @field_validator("scales", "degrees", "translation", mode="after")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.field_name
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field_name)
        return v


class RandomElasticDeformationConfig(TransformConfig):
    """Config class for RandomElasticDeformation transform."""

    num_control_points: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    max_displacement: Union[
        NonNegativeFloat,
        Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    locked_borders: Union[
        LockedBordersMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    image_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    label_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_DEFORMATION

    @field_validator("num_control_points", mode="after")
    @classmethod
    def validator_num_control_points(cls, v):
        """Checks that 'num_control_points' is more than 4."""
        if isinstance(v, int) and v < 4:
            raise ValueError(f"'num_control_points' must be at least 4. Got {v}")
        if isinstance(v, tuple):
            for v_ in v:
                if v_ < 4:
                    raise ValueError(
                        f"'num_control_points' must be at least 4. Got {v_}"
                    )
        return v


class RandomAnisotropyConfig(TransformConfig):
    """Config class for RandomAnisotropy transform."""

    axes: Union[
        NumericalAxis, Tuple[NumericalAxis, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    downsampling: Union[
        PositiveFloat, Tuple[PositiveFloat, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    image_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_ANISOTROPY

    @field_validator("downsampling", mode="after")
    @classmethod
    def validator_downsampling(cls, v):
        """Checks that 'downsampling' values are greater than 1, and sorted if tuple."""
        if isinstance(v, float) and v < 1:
            raise ValueError(
                f"'downsampling' values must be greater or equal to 1. Got {v}"
            )
        elif isinstance(v, tuple):
            cls._check_spatial_tuple(v, "downsampling")
            for v_ in v:
                if v_ < 1:
                    raise ValueError(
                        f"'downsampling' values must be greater or equal to 1. Got {v}"
                    )
        return v
