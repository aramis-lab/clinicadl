from typing import Tuple, Union

from pydantic import PositiveInt, computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import Bounds, ImplementedTransform, TransformConfig
from .enum import EnsureShapeMultipleMode, InterpolationMode, PaddingMode

__all__ = ["ResizeConfig", "EnsureShapeMultipleConfig", "CropConfig", "PadConfig"]


class ResizeConfig(TransformConfig):
    """Config class for Resize transform."""

    target_shape: Union[int, Tuple[int, int, int]]
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
        return ImplementedTransform.RESIZE

    @field_validator("target_shape", mode="after")
    @classmethod
    def validator_target_shape(cls, v):
        """Checks that 'target_shape' contains positive integers (or -1)."""
        if isinstance(v, int):
            cls._check_dimension(v)
        elif isinstance(v, tuple):
            for v_ in v:
                cls._check_dimension(v_)
        return v

    @staticmethod
    def _check_dimension(dim: int) -> None:
        """Checks that the value given for a dimension is either -1 or a positive integer."""
        if (dim <= 0) and (dim != -1):
            raise ValueError(
                "The size of dimensions passed in 'target_shape' must be positive "
                f"integers or -1. Got {dim}"
            )


class EnsureShapeMultipleConfig(TransformConfig):
    """Config class for EnsureShapeMultiple transform."""

    target_multiple: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    method: Union[EnsureShapeMultipleMode, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.ENSURE_MULTIPLE


class CropConfig(TransformConfig):
    """Config class for Crop transform."""

    cropping: Union[Bounds, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.CROP


class PadConfig(TransformConfig):
    """Config class for Pad transform."""

    padding: Union[Bounds, DefaultFromLibrary] = DefaultFromLibrary.YES
    padding_mode: Union[float, PaddingMode, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.PAD
