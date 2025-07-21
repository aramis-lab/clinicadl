from typing import Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

from clinicadl.utils.factories import get_defaults_from

from ..types import SpatialRange
from .base import TorchioTransformConfig
from .enum import (
    AnatomicalAxis,
    CenterMode,
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

RANDOM_FLIP_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomFlip)
RANDOM_AFFINE_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomAffine)
RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS = get_defaults_from(
    tio.transforms.RandomElasticDeformation
)
RANDOM_ANISOTROPY_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.RandomAnisotropy)


class RandomFlipConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomFlip`.
    """

    axes: Union[
        NumericalAxis,
        Tuple[NumericalAxis, ...],
        AnatomicalAxis,
        Tuple[AnatomicalAxis, ...],
    ] = RANDOM_FLIP_TORCHIO_DEFAULTS["axes"]
    flip_probability: float = RANDOM_FLIP_TORCHIO_DEFAULTS["flip_probability"]

    @field_validator("flip_probability", mode="after")
    @classmethod
    def validator_flip_probability(cls, v):
        """Checks that 'flip_probability' is a probability."""
        if isinstance(v, float) and not (0 <= v <= 1):
            raise ValueError("'flip_probability' must be between 0 and 1.")
        return v


class RandomAffineConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomAffine`.
    """

    scales: SpatialRange = RANDOM_AFFINE_TORCHIO_DEFAULTS["scales"]
    degrees: SpatialRange = RANDOM_AFFINE_TORCHIO_DEFAULTS["degrees"]
    translation: SpatialRange = RANDOM_AFFINE_TORCHIO_DEFAULTS["translation"]
    isotropic: bool = RANDOM_AFFINE_TORCHIO_DEFAULTS["isotropic"]
    center: CenterMode = RANDOM_AFFINE_TORCHIO_DEFAULTS["center"]
    default_pad_value: Union[
        float, RandomAffinePaddingMode
    ] = RANDOM_AFFINE_TORCHIO_DEFAULTS["default_pad_value"]
    image_interpolation: InterpolationMode = RANDOM_AFFINE_TORCHIO_DEFAULTS[
        "image_interpolation"
    ]
    label_interpolation: InterpolationMode = RANDOM_AFFINE_TORCHIO_DEFAULTS[
        "label_interpolation"
    ]
    check_shape: bool = RANDOM_AFFINE_TORCHIO_DEFAULTS["check_shape"]

    @field_validator("scales", "degrees", "translation", mode="after")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.field_name
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field_name)
        return v


class RandomElasticDeformationConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomElasticDeformation`.
    """

    num_control_points: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]
    ] = RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS["num_control_points"]
    max_displacement: Union[
        NonNegativeFloat,
        Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    ] = RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS["max_displacement"]
    locked_borders: LockedBordersMode = RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS[
        "locked_borders"
    ]
    image_interpolation: InterpolationMode = (
        RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS["image_interpolation"]
    )
    label_interpolation: InterpolationMode = (
        RANDOM_ELASTIC_DEFORMATION_TORCHIO_DEFAULTS["label_interpolation"]
    )

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


class RandomAnisotropyConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.RandomAnisotropy`.
    """

    axes: Union[
        NumericalAxis, Tuple[NumericalAxis, ...]
    ] = RANDOM_ANISOTROPY_TORCHIO_DEFAULTS["axes"]
    downsampling: Union[
        PositiveFloat, Tuple[PositiveFloat, PositiveFloat]
    ] = RANDOM_ANISOTROPY_TORCHIO_DEFAULTS["downsampling"]
    image_interpolation: InterpolationMode = RANDOM_ANISOTROPY_TORCHIO_DEFAULTS[
        "image_interpolation"
    ]

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
