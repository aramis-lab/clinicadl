from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig
from .enum import Kernel

__all__ = [
    "PSNRConfig",
    "SSIMConfig",
    "MultiScaleSSIMConfig",
]


class PSNRConfig(MetricConfig):
    "Config class for PSNR."

    max_val: PositiveFloat

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "PSNRMetric"


class SSIMConfig(MetricConfig):
    "Config class for SSIM."

    spatial_dims: PositiveInt
    data_range: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    kernel_type: Union[Kernel, DefaultFromLibrary] = DefaultFromLibrary.YES
    win_size: Union[
        PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    kernel_sigma: Union[
        PositiveFloat, Tuple[PositiveFloat, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    k1: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    k2: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "SSIMMetric"

    @field_validator("spatial_dims", mode="after")
    @classmethod
    def validator_spatial_dims(cls, v):
        assert v == 2 or v == 3, f"spatial_dims must be 2 or 3. You passed: {v}."

        return v

    @model_validator(mode="after")
    def dimension_validator(self):
        """Checks coherence between fields."""
        self._check_spatial_dim("win_size")
        self._check_spatial_dim("kernel_sigma")

        return self

    def _check_spatial_dim(self, attribute: str) -> None:
        """Checks that the dimensionality of an attribute is consistent with self.spatial_dims."""
        value = getattr(self, attribute)
        if isinstance(value, tuple):
            assert (
                len(value) == self.spatial_dims
            ), f"If you pass a sequence for {attribute}, it must be of size {self.spatial_dims}. You passed: {value}."


class MultiScaleSSIMConfig(SSIMConfig):
    "Config class for multi-scale SSIM."

    weights: Union[
        Tuple[PositiveFloat, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def metric(self) -> str:
        """The name of the metric."""
        return "MultiScaleSSIMMetric"
