from __future__ import annotations

from typing import Optional, Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from clinicadl.utils.factories import get_defaults_from

from ..types import Std
from .base import TorchioTransformConfig
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


RANDOM_MOTION_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomMotion)
RANDOM_GHOSTING_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomGhosting)
RANDOM_SPIKE_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomSpike)
RANDOM_BIAS_FIELD_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomBiasField)
RANDOM_BLUR_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomBlur)
RANDOM_NOISE_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomNoise)
RANDOOM_GAMMA_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomGamma)
RANOM_SWAP_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomSwap)


class RandomMotionConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomMotion`.
    """

    degrees: Union[
        NonNegativeFloat, tuple[float, float]
    ] = RANDOM_MOTION_TORCHIO_DEFAULTS["degrees"]
    translation: Union[
        NonNegativeFloat, Tuple[float, float]
    ] = RANDOM_MOTION_TORCHIO_DEFAULTS["translation"]
    num_transforms: PositiveInt = RANDOM_MOTION_TORCHIO_DEFAULTS["num_transforms"]
    image_interpolation: InterpolationMode = RANDOM_MOTION_TORCHIO_DEFAULTS[
        "image_interpolation"
    ]

    @field_validator("degrees", "translation", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomGhostingConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomGhosting`.
    """

    num_ghosts: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]
    ] = RANDOM_GHOSTING_TORCHIO_DEFAULTS["num_ghosts"]
    axes: Union[
        NumericalAxis, Tuple[NumericalAxis, ...]
    ] = RANDOM_GHOSTING_TORCHIO_DEFAULTS["axes"]
    intensity: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]
    ] = RANDOM_GHOSTING_TORCHIO_DEFAULTS["intensity"]
    restore: Optional[
        Union[NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]]
    ] = RANDOM_GHOSTING_TORCHIO_DEFAULTS["restore"]

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
        """Checks that 'restore' contains probability."""
        if isinstance(v, float):
            cls._check_restore(v)
        elif isinstance(v, tuple):
            for v_ in v:
                cls._check_restore(v_)
        return v

    @staticmethod
    def _check_restore(restore: float) -> None:
        """Checks a single restore value."""
        if not (0 <= restore <= 1):
            raise ValueError(
                f"'restore' must contain values between 0 and 1. Got {restore}"
            )


class RandomSpikeConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomSpike`.
    """

    num_spikes: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]
    ] = RANDOM_SPIKE_TORCHIO_DEFAULTS["num_spikes"]
    intensity: Union[
        NonNegativeFloat, Tuple[float, float]
    ] = RANDOM_SPIKE_TORCHIO_DEFAULTS["intensity"]

    @field_validator("num_spikes", "intensity", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomBiasFieldConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomBiasField`.
    """

    coefficients: Union[
        NonNegativeFloat, Tuple[float, float]
    ] = RANDOM_BIAS_FIELD_TORCHIO_DEFAULTS["coefficients"]
    order: NonNegativeInt = RANDOM_BIAS_FIELD_TORCHIO_DEFAULTS["order"]

    @field_validator("coefficients", mode="after")
    @classmethod
    def validator_coefficients(cls, v):
        """Checks that 'coefficients' is sorted if tuple."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, "coefficients")
        return v


class RandomBlurConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomBlur`.
    """

    std: Std = RANDOM_BLUR_TORCHIO_DEFAULTS["std"]

    @field_validator("std", mode="after")
    @classmethod
    def validator_std(cls, v):
        """Checks that 'std' is sorted in each dimension."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, "std")
        return v


class RandomNoiseConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomNoise`.
    """

    mean: Union[NonNegativeFloat, Tuple[float, float]] = RANDOM_NOISE_TORCHIO_DEFAULTS[
        "mean"
    ]
    std: Union[
        NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]
    ] = RANDOM_NOISE_TORCHIO_DEFAULTS["std"]

    @field_validator("mean", "std", mode="after")
    @classmethod
    def validate_tuples(cls, v, field):
        """Checks that tuples are ordered."""
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field.field_name)
        return v


class RandomSwapConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomSwap`.
    """

    patch_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]
    ] = RANOM_SWAP_TORCHIO_DEFAULTS["patch_size"]
    num_iterations: NonNegativeInt = RANOM_SWAP_TORCHIO_DEFAULTS["num_iterations"]


class RandomGammaConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomGamma`.
    """

    log_gamma: Union[
        NonNegativeFloat, Tuple[float, float]
    ] = RANDOOM_GAMMA_TORCHIO_DEFAULTS["log_gamma"]
