from typing import Callable, Optional, Sequence, Union

import torch
from monai import transforms
from monai.transforms import Transform as MonaiTransform
from pydantic import (
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.dictionary.words import EXCLUDE, INCLUDE, NAME
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper
from clinicadl.utils.factories import get_defaults_from

from ..homemade import Format
from ..types import Transform
from .base import TransformConfig
from .enum import Rounding, SobelPaddingMode

__all__ = [
    "ActivationsConfig",
    "AsDiscreteConfig",
    "KeepLargestConnectedComponentConfig",
    "DistanceTransformEDTConfig",
    "RemoveSmallObjectsConfig",
    "LabelFilterConfig",
    "FillHolesConfig",
    "SobelGradientsConfig",
    "FormatConfig",
]

ACTIVATIONS_MONAI_DEFAULTS = get_defaults_from(transforms.Activations)
AS_DISCRETE_MONAI_DEFAULTS = get_defaults_from(transforms.AsDiscrete)
KLCC_MONAI_DEFAULTS = get_defaults_from(transforms.KeepLargestConnectedComponent)
EDT_MONAI_DEFAULTS = get_defaults_from(transforms.DistanceTransformEDT)
SMALL_OBJECTS_MONAI_DEFAULTS = get_defaults_from(transforms.RemoveSmallObjects)
LABEL_FILTER_MONAI_DEFAULTS = get_defaults_from(transforms.LabelFilter)
FILL_HOLES_MONAI_DEFAULTS = get_defaults_from(transforms.FillHoles)
SOBEL_MONAI_DEFAULTS = get_defaults_from(transforms.SobelGradients)
FORMAT_DEFAULTS = get_defaults_from(Format)


class MonaiTransformConfig(TransformConfig):
    """
    Base config class for MONAI Transforms.
    """

    def get_object(self) -> Transform:
        """
        Returns the transform associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Transform:
            The associated transform.
        """
        monai_transform = self._get_class()(
            **self.model_dump(exclude={NAME, INCLUDE, EXCLUDE})
        )
        transform = MonaiTransformWrapper(
            monai_transform, include=self.include, exclude=self.exclude
        )
        return transform

    @classmethod
    def _get_class(cls) -> type[MonaiTransform]:
        """Returns the transform associated to this config class."""
        return getattr(transforms, cls._get_name())


class ActivationsConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.Activations`.
    """

    sigmoid: bool = ACTIVATIONS_MONAI_DEFAULTS["sigmoid"]
    softmax: bool = ACTIVATIONS_MONAI_DEFAULTS["softmax"]
    other: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = ACTIVATIONS_MONAI_DEFAULTS["other"]

    @model_validator(mode="after")
    def exclude_multiple_arguments(self):
        """Ensure that the user pass only one argument."""
        arguments = [self.sigmoid, self.softmax, self.other]
        count = sum(1 for item in arguments if item is not None and item is not False)
        if count > 1:
            raise ValueError(
                "You cannot pass more than one argument 'ActivationsConfig'."
            )
        elif count == 0:
            raise ValueError("Please pass at least on argument to 'ActivationsConfig'.")

        return self


class AsDiscreteConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.AsDiscrete`.
    """

    argmax: bool = AS_DISCRETE_MONAI_DEFAULTS["argmax"]
    to_onehot: Optional[PositiveInt] = AS_DISCRETE_MONAI_DEFAULTS["to_onehot"]
    threshold: Optional[float] = AS_DISCRETE_MONAI_DEFAULTS["threshold"]
    rounding: Optional[Rounding] = AS_DISCRETE_MONAI_DEFAULTS["rounding"]
    dtype: torch.dtype = torch.float

    @model_validator(mode="after")
    def exclude_multiple_arguments(self):
        """Ensure that the user pass only one argument."""
        arguments = [self.argmax, self.to_onehot, self.threshold, self.rounding]
        count = sum(1 for item in arguments if item is not None and item is not False)
        if count > 1:
            raise ValueError(
                "You cannot pass more than one argument 'AsDiscreteConfig'."
            )
        elif count == 0:
            raise ValueError("Please pass at least on argument to 'AsDiscreteConfig'.")

        return self


class KeepLargestConnectedComponentConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.KeepLargestConnectedComponent`.
    """

    applied_labels: Optional[Union[int, list[int]]] = KLCC_MONAI_DEFAULTS[
        "applied_labels"
    ]
    is_onehot: Optional[bool] = KLCC_MONAI_DEFAULTS["is_onehot"]
    independent: bool = KLCC_MONAI_DEFAULTS["independent"]
    connectivity: Optional[PositiveInt] = KLCC_MONAI_DEFAULTS["connectivity"]
    num_components: Optional[PositiveInt] = KLCC_MONAI_DEFAULTS["num_components"]


class DistanceTransformEDTConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.DistanceTransformEDT`.
    """

    sampling: Optional[Union[float, list[float]]] = EDT_MONAI_DEFAULTS["sampling"]


class RemoveSmallObjectsConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.RemoveSmallObjects`.
    """

    min_size: PositiveInt = SMALL_OBJECTS_MONAI_DEFAULTS["min_size"]
    connectivity: Optional[PositiveInt] = SMALL_OBJECTS_MONAI_DEFAULTS["connectivity"]
    independent_channels: bool = SMALL_OBJECTS_MONAI_DEFAULTS["independent_channels"]
    by_measure: bool = SMALL_OBJECTS_MONAI_DEFAULTS["by_measure"]
    pixdim: Optional[
        Union[PositiveFloat, list[PositiveFloat]]
    ] = SMALL_OBJECTS_MONAI_DEFAULTS["pixdim"]


class LabelFilterConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.LabelFilter`.
    """

    applied_labels: Union[int, list[int]]


class FillHolesConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.FillHoles`.
    """

    applied_labels: Optional[Union[int, list[int]]] = FILL_HOLES_MONAI_DEFAULTS[
        "applied_labels"
    ]
    connectivity: Optional[PositiveInt] = FILL_HOLES_MONAI_DEFAULTS["connectivity"]


class SobelGradientsConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`monai.transforms.SobelGradients`.
    """

    kernel_size: PositiveInt = SOBEL_MONAI_DEFAULTS["kernel_size"]
    spatial_axes: Optional[
        Union[NonNegativeInt, list[NonNegativeInt]]
    ] = SOBEL_MONAI_DEFAULTS["spatial_axes"]
    normalize_kernels: bool = SOBEL_MONAI_DEFAULTS["normalize_kernels"]
    normalize_gradients: bool = SOBEL_MONAI_DEFAULTS["normalize_gradients"]
    padding_mode: SobelPaddingMode = SOBEL_MONAI_DEFAULTS["padding_mode"]
    dtype: torch.dtype = SOBEL_MONAI_DEFAULTS["dtype"]

    @field_validator("kernel_size", mode="after")
    @classmethod
    def kernel_size_validator(cls, v):
        """'kernel_size' should be odd and more than 3."""
        if v < 3:
            raise ValueError(f"'kernel_size' should be at least 3. Got {v}")
        if v % 2 == 0:
            raise ValueError(f"'kernel_size' should be odd. Got {v}")
        return v


class FormatConfig(MonaiTransformConfig):
    """
    Config class for :py:class:`clinicadl.transforms.homemade.Format`.
    """

    dtype: Optional[torch.dtype] = FORMAT_DEFAULTS["dtype"]
    squeeze: Union[bool, NonNegativeInt, Sequence[NonNegativeInt]] = FORMAT_DEFAULTS[
        "squeeze"
    ]
    unsqueeze: Optional[NonNegativeInt] = FORMAT_DEFAULTS["unsqueeze"]

    @classmethod
    def _get_class(cls) -> type[MonaiTransform]:
        """Returns the transform associated to this config class."""
        return Format
