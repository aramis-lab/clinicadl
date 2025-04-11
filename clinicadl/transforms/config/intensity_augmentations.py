from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from clinicadl.utils.config import DefaultFromLibrary

from .base import TransformConfig
from .enum import InterpolationMode, NumericalAxis

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
    """
    Config class for :py:class:`torchio.transforms.RandomMotion`.
    """

    degrees: Union[NonNegativeFloat, Tuple[float, float]]
    translation: Union[NonNegativeFloat, Tuple[float, float]]
    num_transforms: PositiveInt
    image_interpolation: InterpolationMode

    def __init__(
        self,
        degrees: Union[NonNegativeFloat, Tuple[float, float], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        translation: Union[
            NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        num_transforms: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            degrees=degrees,
            translation=translation,
            num_transforms=num_transforms,
            image_interpolation=image_interpolation,
        )

    @field_validator("degrees", "translation", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomGhostingConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomGhosting`.
    """

    num_ghosts: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
    axes: Union[NumericalAxis, Tuple[NumericalAxis, ...]]
    intensity: Union[NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]]
    restore: NonNegativeFloat

    def __init__(
        self,
        num_ghosts: Union[
            NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        axes: Union[NumericalAxis, Tuple[NumericalAxis, ...], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        intensity: Union[
            NonNegativeFloat,
            Tuple[NonNegativeFloat, NonNegativeFloat],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        restore: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            num_ghosts=num_ghosts,
            axes=axes,
            intensity=intensity,
            restore=restore,
        )

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
        return v


class RandomSpikeConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomSpike`.
    """

    num_spikes: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
    intensity: Union[NonNegativeFloat, Tuple[float, float]]

    def __init__(
        self,
        num_spikes: Union[
            NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        intensity: Union[NonNegativeFloat, Tuple[float, float], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            num_spikes=num_spikes,
            intensity=intensity,
        )

    @field_validator("num_spikes", "intensity", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomBiasFieldConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomBiasField`.
    """

    coefficients: Union[NonNegativeFloat, Tuple[float, float]]
    order: NonNegativeInt

    def __init__(
        self,
        coefficients: Union[
            NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        order: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            coefficients=coefficients,
            order=order,
        )

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
    """
    Config class for :py:class:`torchio.transforms.RandomBlur`.
    """

    std: Std

    def __init__(self, std: Union[Std, DefaultFromLibrary] = DefaultFromLibrary.YES):
        super().__init__(std=std)

    @field_validator("std", mode="after")
    @classmethod
    def validator_std(cls, v):
        """Checks that 'std' is sorted in each dimension."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, "std")
        return v


class RandomNoiseConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomNoise`.
    """

    mean: Union[NonNegativeFloat, Tuple[float, float]]
    std: Union[NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]]

    def __init__(
        self,
        mean: Union[NonNegativeFloat, Tuple[float, float], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        std: Union[
            NonNegativeFloat,
            Tuple[NonNegativeFloat, NonNegativeFloat],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(mean=mean, std=std)

    @field_validator("mean", "std", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomSwapConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomSwap`.
    """

    patch_size: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    num_iterations: NonNegativeInt

    def __init__(
        self,
        patch_size: Union[
            PositiveInt,
            Tuple[PositiveInt, PositiveInt, PositiveInt],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        num_iterations: Union[
            NonNegativeInt, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(patch_size=patch_size, num_iterations=num_iterations)


class RandomGammaConfig(TransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomGamma`.
    """

    log_gamma: Union[NonNegativeFloat, Tuple[float, float]]

    def __init__(
        self,
        log_gamma: Union[NonNegativeFloat, Tuple[float, float], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(log_gamma=log_gamma)
