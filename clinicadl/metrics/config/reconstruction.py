from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.losses.enum import Reduction
from clinicadl.utils.factories import DefaultFromLibrary

from .base import MetricConfig, _GetNotNansConfig, _ReductionConfig
from .enum import Kernel, Optimum

__all__ = [
    "PSNRMetricConfig",
    "SSIMMetricConfig",
    "MultiScaleSSIMMetricConfig",
]


class PSNRMetricConfig(MetricConfig, _ReductionConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.PSNRMetric`.
    """

    max_val: PositiveFloat

    def __init__(
        self,
        max_val: PositiveFloat,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Peak Signal-to-Noise Ratio (PSNR) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#monai.metrics.PSNRMetric
        """
        super().__init__(
            max_val=max_val,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX


class _BaseSSIMConfig(_ReductionConfig, _GetNotNansConfig):
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
    Config class for :py:class:`monai.metrics.SSIMMetric`.
    """

    win_size: Union[PositiveInt, Tuple[PositiveInt, ...]]

    def __init__(
        self,
        spatial_dims: PositiveInt,
        data_range: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        kernel_type: Union[Kernel, DefaultFromLibrary] = DefaultFromLibrary.YES,
        win_size: Union[PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        kernel_sigma: Union[
            PositiveFloat, Tuple[PositiveFloat, ...], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        k1: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        k2: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Structural Similarity Index Measure (SSIM) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#structural-similarity-index-measure
        """
        super().__init__(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            win_size=win_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

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

    kernel_size: Union[PositiveInt, Tuple[PositiveInt, ...]]
    weights: Tuple[PositiveFloat, ...]

    def __init__(
        self,
        spatial_dims: PositiveInt,
        data_range: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        kernel_type: Union[Kernel, DefaultFromLibrary] = DefaultFromLibrary.YES,
        kernel_size: Union[PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        kernel_sigma: Union[
            PositiveFloat, Tuple[PositiveFloat, ...], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        k1: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        k2: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        weights: Union[Tuple[PositiveFloat, ...], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES,
        get_not_nans: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        """
        Config class for the Multi-Scale Structural Similarity Index Measure (MS-SSIM) metric. \n
        More info: https://docs.monai.io/en/latest/metrics.html#multi-scale-structural-similarity-index-measure
        """
        super().__init__(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            weights=weights,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MAX

    @model_validator(mode="after")
    def validator_kernel_size(self):
        """Checks coherence between fields."""
        self._check_spatial_dim("kernel_size")

        return self
