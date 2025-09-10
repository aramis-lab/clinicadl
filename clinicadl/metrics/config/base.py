from abc import abstractmethod
from logging import getLogger
from typing import Any, Dict, Optional

import monai.metrics
from pydantic import field_validator, model_validator

from clinicadl.dictionary.words import LABEL, NAME, OUTPUT
from clinicadl.losses.enum import Reduction
from clinicadl.losses.types import Loss
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig
from clinicadl.utils.factories import get_defaults_from

from ..base import Metric
from ..enum import Optimum
from ..monai_wrapper import MonaiMetricWrapper

__all__ = ["MetricConfig", "LossMetricConfig"]

logger = getLogger("clinicadl.metrics")

LOSS_METRIC_MONAI_DEFAULTS = get_defaults_from(monai.metrics.LossMetric)


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


class LossMetricConfig(MetricConfig):
    "Config class to use the loss as a metric."

    loss_fn: Loss
    reduction: Reduction = LOSS_METRIC_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN

    @model_validator(mode="after")
    def check_reduction(self):
        """Removes the reduction of the loss, and add it at the metric level."""
        try:
            loss_reduction = getattr(self.loss_fn, "reduction")
        except AttributeError:
            pass
        else:
            if self.reduction != loss_reduction:
                logger.warning(
                    f"The loss ({type(self.loss_fn).__name__}) has '{loss_reduction}' reduction, "
                    f"whereas in the metric associated to the loss you passed '{self.reduction}' reduction."
                )
            setattr(self.loss_fn, "reduction", "none")

        return self

    def to_dict(self) -> Dict[str, Any]:
        from clinicadl.utils.json import serialize_callable

        my_dict = super().to_dict()
        my_dict["loss_fn"] = serialize_callable(self.loss_fn)

        return my_dict
