from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torchio as tio
from pydantic import (
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)
from torchio import Image

from clinicadl.utils.config import DefaultFromLibrary

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
    """
    Config class for TorchIO's `CropOrPad <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.CropOrPad>`_
    transform.
    """

    target_shape: Optional[
        Union[
            PositiveInt,
            Tuple[PositiveInt, PositiveInt, PositiveInt],
        ]
    ]
    padding_mode: Union[float, PaddingMode]
    mask_name: Optional[str]
    labels: Optional[Tuple[int, ...]]

    def __init__(
        self,
        target_shape: Optional[
            Union[
                PositiveInt,
                Tuple[PositiveInt, PositiveInt, PositiveInt],
                DefaultFromLibrary,
            ]
        ] = DefaultFromLibrary.YES,
        padding_mode: Union[
            float, PaddingMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        mask_name: Optional[Union[str, DefaultFromLibrary]] = DefaultFromLibrary.YES,
        labels: Union[
            Optional[Tuple[int, ...]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.CROP_OR_PAD.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.CropOrPad

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


class ToCanonicalConfig(TransformConfig):
    """
    Config class for TorchIO's `ToCanonical <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.ToCanonical>`_
    transform.
    """

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.TO_CANONICAL.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.ToCanonical


class ResizeConfig(TransformConfig):
    """
    Config class for TorchIO's `Resize <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Resize>`_
    transform.
    """

    target_shape: Union[int, Tuple[int, int, int]]
    image_interpolation: InterpolationMode
    label_interpolation: InterpolationMode

    def __init__(
        self,
        target_shape: Union[int, Tuple[int, int, int]],
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        label_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            target_shape=target_shape,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RESIZE.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Resize

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
    """
    Config class for TorchIO's `Resample <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Resample>`_
    transform.
    """

    target: Union[
        PositiveFloat,
        Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
        str,
        Path,
        Tuple[Tuple[PositiveInt, PositiveInt, PositiveInt], np.ndarray],
    ]
    pre_affine_name: Optional[str] = None
    image_interpolation: InterpolationMode
    label_interpolation: InterpolationMode
    scalars_only: bool

    def __init__(
        self,
        target: Union[
            PositiveFloat,
            Tuple[PositiveFloat, PositiveFloat, PositiveFloat],
            str,
            Path,
            Tuple[Tuple[PositiveInt, PositiveInt, PositiveInt], np.ndarray],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        pre_affine_name: Optional[DefaultFromLibrary] = DefaultFromLibrary.YES,
        image_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        label_interpolation: Union[InterpolationMode, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        scalars_only: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            target=target,
            pre_affine_name=pre_affine_name,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            scalars_only=scalars_only,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RESAMPLE.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Resample

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


class EnsureShapeMultipleConfig(TransformConfig):
    """
    Config class for TorchIO's `EnsureShapeMultiple <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.EnsureShapeMultiple>`_
    transform.
    """

    target_multiple: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    method: EnsureShapeMultipleMode

    def __init__(
        self,
        target_multiple: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]
        ],
        method: Union[
            EnsureShapeMultipleMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            target_multiple=target_multiple,
            method=method,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.ENSURE_MULTIPLE.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.EnsureShapeMultiple


class CropConfig(TransformConfig):
    """
    Config class for TorchIO's `Crop <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Crop>`_
    transform.
    """

    cropping: Bounds

    def __init__(self, cropping: Bounds):
        super().__init__(
            cropping=cropping,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.CROP.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Crop


class PadConfig(TransformConfig):
    """
    Config class for TorchIO's `Pad <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Pad>`_
    transform.
    """

    padding: Bounds
    padding_mode: Union[float, PaddingMode]

    def __init__(
        self,
        padding: Bounds,
        padding_mode: Union[
            float, PaddingMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            padding=padding,
            padding_mode=padding_mode,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.PAD.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Pad
