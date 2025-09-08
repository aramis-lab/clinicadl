from typing import Tuple, Union

import monai
import monai.metrics
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import get_defaults_from

from ..enum import Optimum
from .base import MetricConfig, _GetNotNansConfig
from .enum import Kernel

__all__ = [
    "PSNRMetricConfig",
    "SSIMMetricConfig",
    "MultiScaleSSIMMetricConfig",
]

PSNR_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.PSNRMetric)
SSIM_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.SSIMMetric)
MULTI_SCALE_SSIM_MONAI_DEFAULTS = get_defaults_from(
    monai.metrics.regression.MultiScaleSSIMMetric
)


class PSNRMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.PSNRMetric`.
    """

    max_val: PositiveFloat
    reduction: Reduction = PSNR_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class _BaseSSIMConfig(_GetNotNansConfig):
    "Base config class for SSIM-related metrics."

    spatial_dims: PositiveInt
    data_range: PositiveFloat
    kernel_type: Kernel
    kernel_sigma: Union[PositiveFloat, Tuple[PositiveFloat, ...]]
    k1: NonNegativeFloat
    k2: NonNegativeFloat

    @field_validator("spatial_dims", mode="after")
    @classmethod
    def validator_spatial_dims(cls, v):
        """Validates the spatial dimensions."""
        assert v == 2 or v == 3, f"spatial_dims must be 2 or 3. You passed: {v}."
        return v

    @model_validator(mode="after")
    def validator_kernel_sigma(self):
        """Checks coherence between fields."""
        self._check_spatial_dim("kernel_sigma")

        return self

    def _check_spatial_dim(self, attribute: str) -> None:
        """Checks that the dimensionality of an attribute is consistent with self.spatial_dims."""
        value = getattr(self, attribute)
        if isinstance(value, tuple):
            assert (
                len(value) == self.spatial_dims
            ), f"If you pass a sequence for {attribute}, it must be of size {self.spatial_dims}. You passed: {value}."


class SSIMMetricConfig(MetricConfig, _BaseSSIMConfig):
    """
    Config class for :py:class:`monai.metrics.regression.SSIMMetric`.
    """

    spatial_dims: PositiveInt
    data_range: PositiveFloat = SSIM_MONAI_DEFAULTS["data_range"]
    kernel_type: Kernel = SSIM_MONAI_DEFAULTS["kernel_type"]
    win_size: Union[PositiveInt, Tuple[PositiveInt, ...]] = SSIM_MONAI_DEFAULTS[
        "win_size"
    ]
    kernel_sigma: Union[PositiveFloat, Tuple[PositiveFloat, ...]] = SSIM_MONAI_DEFAULTS[
        "kernel_sigma"
    ]
    k1: NonNegativeFloat = SSIM_MONAI_DEFAULTS["k1"]
    k2: NonNegativeFloat = SSIM_MONAI_DEFAULTS["k2"]
    reduction: Reduction = SSIM_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX

    @model_validator(mode="after")
    def validator_win_size(self):
        """Checks coherence between fields."""
        self._check_spatial_dim("win_size")

        return self


class MultiScaleSSIMMetricConfig(MetricConfig, _BaseSSIMConfig):
    """
    Config class for :py:class:`monai.metrics.MultiScaleSSIMMetric`.
    """

    spatial_dims: PositiveInt
    data_range: PositiveFloat = MULTI_SCALE_SSIM_MONAI_DEFAULTS["data_range"]
    kernel_type: Kernel = MULTI_SCALE_SSIM_MONAI_DEFAULTS["kernel_type"]
    kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, ...]
    ] = MULTI_SCALE_SSIM_MONAI_DEFAULTS["kernel_size"]
    kernel_sigma: Union[
        PositiveFloat, Tuple[PositiveFloat, ...]
    ] = MULTI_SCALE_SSIM_MONAI_DEFAULTS["kernel_sigma"]
    k1: NonNegativeFloat = MULTI_SCALE_SSIM_MONAI_DEFAULTS["k1"]
    k2: NonNegativeFloat = MULTI_SCALE_SSIM_MONAI_DEFAULTS["k2"]
    weights: Tuple[PositiveFloat, ...] = MULTI_SCALE_SSIM_MONAI_DEFAULTS["weights"]
    reduction: Reduction = MULTI_SCALE_SSIM_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX

    @model_validator(mode="after")
    def validator_kernel_size(self):
        """Checks coherence between fields."""
        self._check_spatial_dim("kernel_size")

        return self
