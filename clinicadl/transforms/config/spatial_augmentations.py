from typing import Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.config import DefaultFromLibrary

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

__all__ = [
    "RandomFlipConfig",
    "RandomAffineConfig",
    "RandomElasticDeformationConfig",
    "RandomAnisotropyConfig",
]


class RandomFlipConfig(TransformConfig):
    """
    Config class for TorchIO's `RandomFlip <https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomFlip>`_
    augmentation.
    """

    axes: Union[
        NumericalAxis,
        Tuple[NumericalAxis, ...],
        AnatomicalAxis,
        Tuple[AnatomicalAxis, ...],
    ]
    flip_probability: float

    def __init__(
        self,
        axes: Union[
            NumericalAxis,
            Tuple[NumericalAxis, ...],
            AnatomicalAxis,
            Tuple[AnatomicalAxis, ...],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        flip_probability: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            axes=axes,
            flip_probability=flip_probability,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_FLIP.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RandomFlip

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
    """
    Config class for TorchIO's `RandomAffine <https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine>`_
    augmentation.
    """

    scales: SpatialRange
    degrees: SpatialRange
    translation: SpatialRange
    isotropic: bool
    center: CenterMode
    default_pad_value: Union[float, RandomAffinePaddingMode]
    image_interpolation: InterpolationMode
    label_interpolation: InterpolationMode
    check_shape: bool

    def __init__(
        self,
        scales: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES,
        degrees: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES,
        translation: Union[SpatialRange, DefaultFromLibrary] = DefaultFromLibrary.YES,
        isotropic: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        center: Union[CenterMode, DefaultFromLibrary] = DefaultFromLibrary.YES,
        default_pad_value: Union[float, RandomAffinePaddingMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        label_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        check_shape: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            scales=scales,
            degrees=degrees,
            translation=translation,
            isotropic=isotropic,
            center=center,
            default_pad_value=default_pad_value,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            check_shape=check_shape,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_AFFINE.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RandomAffine

    @field_validator("scales", "degrees", "translation", mode="after")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.field_name
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field_name)
        return v


class RandomElasticDeformationConfig(TransformConfig):
    """
    Config class for TorchIO's `RandomElasticDeformation <https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomElasticDeformation>`_
    augmentation.
    """

    num_control_points: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    max_displacement: Union[
        NonNegativeFloat,
        Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    ]
    locked_borders: LockedBordersMode
    image_interpolation: InterpolationMode
    label_interpolation: InterpolationMode

    def __init__(
        self,
        num_control_points: Union[
            PositiveInt,
            Tuple[PositiveInt, PositiveInt, PositiveInt],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        max_displacement: Union[
            NonNegativeFloat,
            Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        locked_borders: Union[LockedBordersMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        label_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            num_control_points=num_control_points,
            max_displacement=max_displacement,
            locked_borders=locked_borders,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_DEFORMATION.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RandomElasticDeformation

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
    """
    Config class for TorchIO's `RandomAnisotropy <https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAnisotropy>`_
    augmentation.
    """

    axes: Union[NumericalAxis, Tuple[NumericalAxis, ...]]
    downsampling: Union[PositiveFloat, Tuple[PositiveFloat, PositiveFloat]]
    image_interpolation: Union[InterpolationMode]

    def __init__(
        self,
        axes: Union[NumericalAxis, Tuple[NumericalAxis, ...], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        downsampling: Union[
            PositiveFloat, Tuple[PositiveFloat, PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            axes=axes,
            downsampling=downsampling,
            image_interpolation=image_interpolation,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RANDOM_ANISOTROPY.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RandomAnisotropy

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
