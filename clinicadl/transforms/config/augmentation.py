from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .base import TransformConfig
from .enum import (
    AnatomicalAxis,
    CenterMode,
    ImplementedTransform,
    InterpolationMode,
    LockedBordersMode,
    NumericalAxis,
    RandomAffinePaddingMode,
)
from .utils import is_sorted

__all__ = [
    "RandomFlipConfig",
    "RandomAffineConfig",
    "RandomElasticDeformationConfig",
    "RandomAnisotropyConfig",
    "RandomMotionConfig",
    "RandomGhostingConfig",
    "RandomSpikeConfig",
]


class RandomFlipConfig(TransformConfig):
    """Config class for RandomFlip transform."""

    axes: Union[
        NumericalAxis,
        Tuple[NumericalAxis, ...],
        AnatomicalAxis,
        Tuple[AnatomicalAxis, ...],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
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

    @field_validator("scales", "degrees", "translation")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.name
        if isinstance(v, tuple):
            if len(tuple) == 2:
                cls._check_range_tuple(v, field_name)
            elif len(tuple) == 6:
                cls._check_range_tuple(v[:2], field_name)
                cls._check_range_tuple(v[2:4], field_name)
                cls._check_range_tuple(v[4:], field_name)
        return v

    @staticmethod
    def _check_range_tuple(range_: Tuple[float, float], field_name: str) -> None:
        """Checks the consistency between the upper and lower bounds of a range."""
        if not is_sorted(range_):
            raise ValueError(
                f"If {field_name} is a couple, the first element must be smaller "
                f"than the second. Got {range_}"
            )


class RandomElasticDeformationConfig(TransformConfig):
    """Config class for RandomElasticDeformation transform."""

    num_control_points: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    max_displacement: Union[
        NonNegativeInt,
        Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
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
            raise ValueError("'num_control_points' must be at least 4.")
        return v


class _NumericalAxesConfig(ClinicaDLConfig):
    """Config class for 'axes' option when it supports only numerical values."""

    axes: Union[
        NumericalAxis, Tuple[NumericalAxis, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class RandomAnisotropyConfig(TransformConfig, _NumericalAxesConfig):
    """Config class for RandomAnisotropy transform."""

    downsampling: Union[
        PositiveFloat, Tuple[PositiveFloat, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    image_interpolation: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_ANISOTROPY

    @field_validator("downsampling", mode="after")
    @classmethod
    def validator_downsampling(cls, v):
        """Checks that 'downsampling' values are greater than 1."""
        if isinstance(v, float) and v < 1:
            raise ValueError(
                f"'downsampling' values must be greater or equal to 1. Got {v}"
            )
        elif isinstance(v, tuple):
            if not is_sorted(v):
                raise ValueError(
                    "If 'downsampling' is passed as a tuple, the first value must be "
                    f"smaller than the second. Got {v}"
                )
            for v_ in v:
                if v_ < 1:
                    raise ValueError(
                        f"'downsampling' values must be greater or equal to 1. Got {v}"
                    )
        return v


class RandomMotionConfig(TransformConfig):
    """Config class for RandomMotion transform."""

    degrees: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    translation: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    num_transforms: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    image_interpolation: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_MOTION

    @field_validator("degrees", "translation", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple) and not is_sorted(v):
            raise ValueError(
                f"If '{field.field_name}' is passed as a tuple, the first value must be "
                f"smaller than the second. Got {v}"
            )
        return v


class RandomGhostingConfig(TransformConfig, _NumericalAxesConfig):
    """Config class for RandomGhosting transform."""

    num_ghosts: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    intensity: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    restore: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_GHOSTING

    @field_validator("num_ghosts", "intensity", "restore", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple) and not is_sorted(v):
            raise ValueError(
                f"If '{field.field_name}' is passed as a tuple, the first value must be "
                f"smaller than the second. Got {v}"
            )
        return v

    @field_validator("restore", mode="after")
    @classmethod
    def validator_restore(cls, v):
        """Checks that 'restore' is a probability."""
        if isinstance(v, float) and v > 1:
            raise ValueError(f"'restore' must be between 0 and 1. Got {v}")
        elif isinstance(v, tuple) and v[1] > 1:
            raise ValueError(f"'restore' must contain values between 0 and 1. Got {v}")
        return v


class RandomSpikeConfig(TransformConfig):
    """Config class for RandomSpike transform."""

    num_spikes: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    intensity: Union[
        NonNegativeInt, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @field_validator("num_spikes", "intensity", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple) and not is_sorted(v):
            raise ValueError(
                f"If '{field.field_name}' is passed as a tuple, the first value must be "
                f"smaller than the second. Got {v}"
            )
        return v
