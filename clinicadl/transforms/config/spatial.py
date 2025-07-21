from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torchio as tio
from pydantic import (
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from torchio import Image

from clinicadl.utils.factories import get_defaults_from

from .base import Bounds, TorchioTransformConfig
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

CROP_OR_PAD_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.CropOrPad)
TO_CANONICAL_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.ToCanonical)
RESIZE_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.Resize)
RESAMPLE_TORCHIO_DEFAULTS = get_defaults_from(tio.transforms.Resample)
ENSURE_SHAPE_MULTIPLE_TORCHIO_DEFAULTS = get_defaults_from(
    tio.transforms.EnsureShapeMultiple
)
CROP_TORCHIO_DEFAULT = get_defaults_from(tio.transforms.Crop)
PAD_TORCHIO_DEFAULT = get_defaults_from(tio.transforms.Pad)


class CropOrPadConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.CropOrPad`.
    """

    target_shape: Optional[
        Union[
            PositiveInt,
            Tuple[PositiveInt, PositiveInt, PositiveInt],
        ]
    ] = CROP_OR_PAD_TORCHIO_DEFAULTS["target_shape"]
    padding_mode: Union[float, PaddingMode] = CROP_OR_PAD_TORCHIO_DEFAULTS[
        "padding_mode"
    ]
    mask_name: Optional[str] = CROP_OR_PAD_TORCHIO_DEFAULTS["mask_name"]
    labels: Optional[Tuple[int, ...]] = CROP_OR_PAD_TORCHIO_DEFAULTS["labels"]

    @model_validator(mode="after")
    def check_shape(self):
        """Checks consistency between 'target_shape', 'mask_name' and 'labels'."""
        if not self.target_shape and not self.mask_name:
            raise ValueError(
                "If 'target_shape' is None or is not passed, a valid 'mask_name' must be passed."
            )
        if not self.mask_name and self.labels:
            raise ValueError(
                "If 'mask_name' is not passed, 'labels' must be left to None."
            )
        return self


class ToCanonicalConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.ToCanonical`.
    """


class ResizeConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.Resize`.
    """

    target_shape: Union[int, Tuple[int, int, int]]
    image_interpolation: InterpolationMode = RESIZE_TORCHIO_DEFAULTS[
        "image_interpolation"
    ]
    label_interpolation: InterpolationMode = RESIZE_TORCHIO_DEFAULTS[
        "label_interpolation"
    ]

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


class ResampleConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.Resample`.
    """

    target: Union[
        PositiveFloat,
        Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
        str,
        Path,
        Tuple[Tuple[PositiveInt, PositiveInt, PositiveInt], np.ndarray],
    ] = RESAMPLE_TORCHIO_DEFAULTS["target"]
    pre_affine_name: Optional[str] = RESAMPLE_TORCHIO_DEFAULTS["pre_affine_name"]
    image_interpolation: InterpolationMode = RESAMPLE_TORCHIO_DEFAULTS[
        "image_interpolation"
    ]
    label_interpolation: InterpolationMode = RESAMPLE_TORCHIO_DEFAULTS[
        "label_interpolation"
    ]
    scalars_only: bool = RESAMPLE_TORCHIO_DEFAULTS["scalars_only"]

    @field_validator("pre_affine_name", mode="before")
    @classmethod
    def validator_pre_affine_name(cls, v):
        """Checks that 'pre_affine_name' is not passed."""
        if v is not None:
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


class EnsureShapeMultipleConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.EnsureShapeMultiple`.
    """

    target_multiple: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    method: EnsureShapeMultipleMode = ENSURE_SHAPE_MULTIPLE_TORCHIO_DEFAULTS["method"]


class CropConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.Crop`.
    """

    cropping: Bounds


class PadConfig(TorchioTransformConfig):
    """
    Config class for :py:class:`torchio.transforms.Pad`.
    """

    padding: Bounds
    padding_mode: Union[float, PaddingMode] = PAD_TORCHIO_DEFAULT["padding_mode"]
