from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import (
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)
from torchio import Image

from clinicadl.utils.factories import DefaultFromLibrary

from .base import Bounds, ImplementedTransform, TransformConfig
from .enum import EnsureShapeMultipleMode, InterpolationMode, PaddingMode

__all__ = [
    "CropOrPadConfig",
    "ToCanonicalConfig",
    "ResizeConfig",
    "ResampleConfig",
    "EnsureShapeMultipleConfig",
    "CropConfig",
    "PadConfig",
]


class CropOrPadConfig(TransformConfig):
    """Config class for CropOrPad transform."""

    target_shape: Optional[
        Union[
            PositiveInt,
            Tuple[PositiveInt, PositiveInt, PositiveInt],
            DefaultFromLibrary,
        ]
    ] = DefaultFromLibrary.YES
    padding_mode: Union[float, PaddingMode, DefaultFromLibrary] = DefaultFromLibrary.YES
    mask_name: Optional[Union[str, DefaultFromLibrary]] = DefaultFromLibrary.YES
    labels: Union[
        Optional[Tuple[int, ...]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.CROP_OR_PAD

    @model_validator(mode="after")
    def check_shape(self):
        """Checks consistency between 'target_shape', 'mask_name' and 'labels'."""
        if (
            self.target_shape is None or self.target_shape == DefaultFromLibrary.YES
        ) and (self.mask_name is None or self.mask_name == DefaultFromLibrary.YES):
            raise ValueError(
                "If 'target_shape' is None or is not passed, a valid 'mask_name' must be passed."
            )
        if (
            self.mask_name is None or self.mask_name == DefaultFromLibrary.YES
        ) and not (self.labels is None or self.labels == DefaultFromLibrary.YES):
            raise ValueError(
                "If 'mask_name' is not passed, 'labels' must be left to None."
            )
        return self


class ToCanonicalConfig(TransformConfig):
    """Config class for ToCanonical transform."""

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.TO_CANONICAL


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


class ResampleConfig(TransformConfig):
    """Config class for Resample transform."""

    target: Union[
        PositiveFloat,
        Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
        str,
        Path,
        Tuple[Tuple[PositiveInt, PositiveInt, PositiveInt], np.ndarray],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    pre_affine_name: Optional[DefaultFromLibrary] = DefaultFromLibrary.YES
    image_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    label_interpolation: Union[
        InterpolationMode, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    scalars_only: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.RESAMPLE

    @field_validator("pre_affine_name", mode="before")
    @classmethod
    def validator_pre_affine_name(cls, v):
        """Checks that 'pre_affine_name' is not passed."""
        if v is not None and v != DefaultFromLibrary.YES:
            raise ValueError("'pre_affine_name' is not supported in ClinicaDL.")
        return v

    @field_validator("target", mode="before")
    @classmethod
    def not_tio_image(cls, v):
        """Checks that 'target' is not a TorchIO Image."""
        if isinstance(v, Image):
            raise ValueError("TorchIO Image not supported for 'target'.")
        return v

    @field_validator("target", mode="after")
    @classmethod
    def validator_target(cls, v):
        """Validates 'target' argument."""
        if isinstance(v, tuple) and len(v) == 2:
            affine: np.ndarray = v[1]
            if affine.shape != (4, 4):
                raise ValueError(
                    "If 'target' is passed as '(spatial_shape, affine)', 'affine' must be "
                    f"a numpy array of shape (4, 4). Got shape {affine.shape}"
                )
        elif isinstance(v, Path) and not v.is_file():
            raise ValueError(f"Got a path for 'target', but {v} is not a valid file.")
        return v


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

    cropping: Union[Bounds, DefaultFromLibrary]

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.CROP


class PadConfig(TransformConfig):
    """Config class for Pad transform."""

    padding: Union[Bounds, DefaultFromLibrary]
    padding_mode: Union[float, PaddingMode, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedTransform:
        """The name of the transform."""
        return ImplementedTransform.PAD
