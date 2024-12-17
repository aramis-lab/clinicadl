from typing import Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import TransformConfig
from .enum import AnatomicalAxis, CenterMode, InterpolationMode, RandomAffinePaddingMode

__all__ = ["RandomFlipConfig", "RandomAffine"]


class RandomFlipConfig(TransformConfig):
    """Config class for RandomFlip transform."""

    axes: Union[
        PositiveInt,
        Tuple[PositiveInt, ...],
        AnatomicalAxis,
        Tuple[AnatomicalAxis, ...],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    flip_probability: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES

    @field_validator("axes", mode="after")
    @classmethod
    def validator_axes(cls, v):
        """Checks integer values in 'axes'."""
        if isinstance(v, int):
            cls._check_axis(v)
        elif isinstance(v, tuple) and isinstance(v[0], int):
            for v_ in v:
                cls._check_axis(v_)
        return v

    @staticmethod
    def _check_axis(axis: int) -> None:
        """Checks that an axis passed as an int is equal to 0, 1 or 2."""
        if axis not in {0, 1, 2}:
            raise ValueError(
                f"If 'axes' are passed with integers, they must be 0, 1 and 2. Got {axis}"
            )

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


class RandomAffine(TransformConfig):
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
        if range_[0] > range_[1]:
            raise ValueError(
                f"If {field_name} is a couple, the first element must be smaller "
                f"than the second. Got {range_}"
            )
