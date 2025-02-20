from typing import Optional, Tuple, Union

import torchio as tio
from pydantic import (
    NonNegativeFloat,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.transforms.homemade_transforms import NanRemoval
from clinicadl.utils.config import DefaultFromLibrary

from .base import Bounds, ImplementedTransform, MaskingMethodConfig, TransformConfig
from .enum import AnatomicalLabel

__all__ = [
    "RescaleIntensityConfig",
    "ZNormalizationConfig",
    "MaskConfig",
    "ClampConfig",
    "NanRemovalConfig",
]


class RescaleIntensityConfig(TransformConfig, MaskingMethodConfig):
    """
    Config class for TorchIO's `RescaleIntensity <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.RescaleIntensity>`_
    transform.
    """

    out_min_max: Union[NonNegativeFloat, Tuple[float, float]]
    percentiles: Union[NonNegativeFloat, Tuple[NonNegativeFloat, NonNegativeFloat]]
    in_min_max: Union[Optional[Union[NonNegativeFloat, Tuple[float, float]]]]

    def __init__(
        self,
        out_min_max: Union[
            NonNegativeFloat, Tuple[float, float], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        percentiles: Union[
            NonNegativeFloat,
            Tuple[NonNegativeFloat, NonNegativeFloat],
            DefaultFromLibrary,
        ] = DefaultFromLibrary.YES,
        in_min_max: Union[
            Optional[Union[NonNegativeFloat, Tuple[float, float]]], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        masking_method: Optional[
            Union[str, AnatomicalLabel, Bounds, DefaultFromLibrary]
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            out_min_max=out_min_max,
            percentiles=percentiles,
            in_min_max=in_min_max,
            masking_method=masking_method,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.RESCALE_INTENSITY.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.RescaleIntensity

    @field_validator("out_min_max", "percentiles", "in_min_max", mode="after")
    @classmethod
    def validator_ranges(cls, v, field):
        """Validates the ranges of uniform distributions."""
        field_name = field.field_name
        if isinstance(v, tuple):
            cls._check_spatial_tuple(v, field_name)
        return v

    @field_validator("percentiles", mode="after")
    @classmethod
    def validator_percentiles(cls, v):
        """Checks that percentiles are between 0 and 100."""
        if isinstance(v, float):
            cls._check_percentile(v)
        elif isinstance(v, tuple):
            cls._check_percentile(v[0])
            cls._check_percentile(v[1])
        return v

    @staticmethod
    def _check_percentile(percentile: float) -> None:
        """Checks a single percentile."""
        if not (0 <= percentile <= 100):
            raise ValueError(
                f"'percentiles' must contain values between 0 and 100. Got {percentile}"
            )


class ZNormalizationConfig(TransformConfig, MaskingMethodConfig):
    """
    Config class for TorchIO's `ZNormalization <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.ZNormalization>`_
    transform.
    """

    def __init__(
        self,
        masking_method: Optional[
            Union[str, AnatomicalLabel, Bounds, DefaultFromLibrary]
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            masking_method=masking_method,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.Z_NORMALIZATION.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.ZNormalization


class MaskConfig(TransformConfig, MaskingMethodConfig):
    """
    Config class for TorchIO's `Mask <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Mask>`_
    transform.
    """

    outside_value: float
    labels: Optional[Tuple[int, ...]]

    def __init__(
        self,
        masking_method: Optional[Union[str, AnatomicalLabel, Bounds]],
        outside_value: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
        labels: Union[Optional[Tuple[int, ...]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            masking_method=masking_method, outside_value=outside_value, labels=labels
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.MASK.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Mask


class ClampConfig(TransformConfig):
    """
    Config class for TorchIO's `Clamp <https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Clamp>`_
    transform.
    """

    out_min: Optional[float]
    out_max: Optional[float]

    def __init__(
        self,
        out_min: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
        out_max: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(out_min=out_min, out_max=out_max)

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.CLAMP.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return tio.Clamp

    @model_validator(mode="after")
    def validate_min_max(self):
        """Checks consistency between 'out_min' and 'out_max'."""
        if not self.out_min and not self.out_max:
            raise ValueError("'out_min' and 'out_max' cannot both be None.")
        elif self.out_min and self.out_max and self.out_min > self.out_max:
            raise ValueError(
                f"'out_min' should be smaller than 'out_max'. Got out_min={self.out_min} and out_max={self.out_max}"
            )

        return self


class NanRemovalConfig(TransformConfig):
    """
    Config class for ClinicaDL's :ref:`nan_removal` transform.
    """

    nan: float
    posinf: Optional[float]
    neginf: Optional[float]

    def __init__(
        self,
        nan: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
        posinf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
        neginf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(nan=nan, posinf=posinf, neginf=neginf)

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ImplementedTransform.NAN_REMOVAL.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return NanRemoval
