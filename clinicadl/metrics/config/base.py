from abc import ABC, abstractmethod
from typing import Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.losses.utils import Loss
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import ImplementedMetric, Reduction

__all__ = ["MetricConfig", "LossMetricConfig"]


class MetricConfig(BaseModel, ABC):
    """Base config class to configure metrics."""

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedMetric:
        """The name of the metric."""


class _MetricWithBackgroundConfig(MetricConfig):
    """Base config class to configure metrics with 'include_background' parameter."""

    include_background: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES


class _MetricWithReductionConfig(MetricConfig):
    """Base config class to configure metrics with 'reduction' parameter."""

    reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES


class _MetricWithNotNansConfig(MetricConfig):
    """Base config class to configure metrics with 'get_not_nans' parameter."""

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
