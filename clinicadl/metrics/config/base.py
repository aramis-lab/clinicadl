from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

import monai
import monai.metrics
from pydantic import field_validator, model_validator

from clinicadl.losses.enum import Reduction
from clinicadl.losses.types import Loss
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig

from .enum import Optimum

__all__ = ["MetricConfig", "LossMetricConfig"]


class MetricConfig(ObjectConfig):
    """Base config class to configure metrics."""

    def get_object(self) -> monai.metrics.metric.CumulativeIterationMetric:
        """
        Returns the metric associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        monai.metrics.Metric:
            The MONAI metric.
        """
        return super().get_object()

    @classmethod
    def _get_class(cls) -> type[monai.metrics.metric.CumulativeIterationMetric]:
        """Returns the metric associated to this config class."""
        return getattr(monai.metrics, cls._get_name())

    @staticmethod
    @abstractmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""


class _GetNotNansConfig(ClinicaDLConfig):
    """Config class for 'get_not_nans' parameter."""

    get_not_nans: bool = False

    @field_validator("get_not_nans", mode="after")
    @classmethod
    def validator_get_not_nans(cls, v):
        assert (
            not v
        ), "'get_not_nans' currently not supported in ClinicaDL. Please leave to False."

        return v


class LossMetricConfig(MetricConfig):
    "Config class to use the loss as a metric."

    loss_fn: Loss
    reduction: Optional[Reduction] = None

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN

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

    def to_dict(self) -> Dict[str, Any]:
        from clinicadl.utils.json import serialize_callable

        my_dict = super().to_dict()
        my_dict["loss_fn"] = serialize_callable(self.loss_fn)

        return my_dict
