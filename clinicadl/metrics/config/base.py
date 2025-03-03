from abc import ABC, abstractmethod
from typing import Optional, Union

from pydantic import (
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.losses.types import Loss
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import ImplementedMetric, Reduction

__all__ = ["MetricConfig", "LossMetricConfig"]


class MetricConfig(ClinicaDLConfig, ABC):
    """Base config class to configure metrics."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedMetric:
        """The name of the metric."""


class _IncludeBackgroundConfig(ClinicaDLConfig):
    """Base config class for 'include_background' parameter."""

    include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES


class _ReductionConfig(ClinicaDLConfig):
    """Base config class for 'reduction' parameter."""

    reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES


class _GetNotNansConfig(ClinicaDLConfig):
    """Base config class for 'get_not_nans' parameter."""

    get_not_nans: bool = False

    @field_validator("get_not_nans", mode="after")
    @classmethod
    def validator_get_not_nans(cls, v):
        assert not v, "'get_not_nans' not supported in ClinicaDL. Please set to False."

        return v


class LossMetricConfig(MetricConfig):
    "Config class to use the loss as a metric."

    loss_fn: Loss
    reduction: Optional[Reduction] = None

    @computed_field
    @property
    def name(self) -> str:
        """The name of the metric."""
        return "LossMetric"

    @model_validator(mode="after")
    def check_reduction(self):
        """If 'reduction' is None, the reduction method of the loss function will be used."""
        if self.reduction is None:
            try:
                self.reduction = self.loss_fn.reduction
            except AttributeError as exc:
                raise ValueError(
                    "If the loss function doesn't have an attribute 'reduction', you must pass a reduction method to "
                    "use the loss as a metric."
                ) from exc

        return self
