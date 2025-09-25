from abc import abstractmethod
from logging import getLogger
from typing import Optional

import monai.metrics
from pydantic import field_validator

from clinicadl.dictionary.words import LABEL, NAME, OUTPUT
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig

from ..base import Metric
from ..enum import Optimum
from ..monai_wrapper import MonaiMetricWrapper

__all__ = ["MetricConfig"]

logger = getLogger("clinicadl.metrics")


class MetricConfig(ObjectConfig):
    """Base config class to configure metrics."""

    pred_key: str = OUTPUT
    label_key: Optional[str] = LABEL
    postprocessing: list[TransformOrConfig] = []

    def get_object(self) -> Metric:
        """
        Returns the metric associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Metric:
            The associated metric.
        """
        monai_metric = self._get_class()(
            **self.model_dump(exclude={NAME, "pred_key", "label_key", "postprocessing"})
        )
        metric = MonaiMetricWrapper(
            monai_metric,
            pred_key=self.pred_key,
            label_key=self.label_key,
            optimum=self.optimum(),
            postprocessing=self.postprocessing,
        )
        return metric

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
