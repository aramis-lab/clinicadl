from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import TransformConfig, _NumericalAxesConfig
from .enum import ImplementedTransform

__all__ = [
    "RandomMotionConfig",
    "RandomGhostingConfig",
    "RandomSpikeConfig",
    "RandomBiasFieldConfig",
    "RandomBlurConfig",
    "RandomNoiseConfig",
    "RandomSwapConfig",
    "RandomGammaConfig",
]


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
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
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
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
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

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_SPIKE

    @field_validator("num_spikes", "intensity", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomBiasFieldConfig(TransformConfig):
    """Config class for RandomBiasField transform."""

    coefficients: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    order: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_BIAS_FIELD

    @field_validator("coefficients", mode="after")
    @classmethod
    def validator_coefficients(cls, v):
        """Checks that 'coefficients' is sorted if tuple."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, "coefficients")
        return v


Std = Union[
    NonNegativeFloat,
    Tuple[NonNegativeFloat, NonNegativeFloat],
    Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    Tuple[
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
    ],
]


class RandomBlurConfig(TransformConfig):
    """Config class for RandomBlur transform."""

    std: Union[Std, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_BLUR

    @field_validator("std", mode="after")
    @classmethod
    def validator_std(cls, v):
        """Checks that 'std' is sorted in each dimension."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, "std")
        return v


class RandomNoiseConfig(TransformConfig):
    """Config class for RandomNoise transform."""

    mean: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    std: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_NOISE

    @field_validator("mean", "std", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomSwapConfig(TransformConfig):
    """Config class for RandomSwap transform."""

    patch_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    num_iterations: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_SWAP


class RandomGammaConfig(TransformConfig):
    """Config class for RandomGamma transform."""

    log_gamma: Union[
        NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_GAMMA
