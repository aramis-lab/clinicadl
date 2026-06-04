from abc import abstractmethod
from logging import getLogger
from typing import Any, Optional, Sequence, Union

import monai.metrics
from pydantic import Field, field_validator

from clinicadl.transforms.handlers import PostprocessingHandler
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig
from clinicadl.utils.dictionary.words import LABEL, OUTPUT

from ..base import Metric
from ..enum import Optimum
from ..monai_wrapper import MonaiMetricWrapper

__all__ = ["MetricConfig"]

logger = getLogger(__name__)

DOCUMENT_EXTRA_PARAMETERS = """
``pred_key`` corresponds to the key of the model output to evaluate
in the :py:class:`~clinicadl.data.structures.DataPoint`; ``label_key`` is
the key of the potential label to which the model output must be compared.

Potential postprocessing to apply to the model output before computing the metric
can be specified via ``postprocessing``. Accepted transforms are functions that take as input a ``DataPoint`` and return
a ``DataPoint``, or :py:mod:`configuration classes <clinicadl.transforms.config>`.
"""


class MetricConfig(ObjectConfig[Metric]):
    """Base config class to configure metrics."""

    pred_key: str = OUTPUT
    label_key: Optional[str] = LABEL
    postprocessing: Union[list[TransformOrConfig], PostprocessingHandler] = Field(
        default=[], reader=PostprocessingHandler.from_dict
    )

    def get_object(self, **kwargs: Any) -> Metric:
        """
        Returns the metric associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Metric:
            The associated metric.
        """
        monai_metric = self._get_class()(
            **self.to_raw_dict(exclude={"pred_key", "label_key", "postprocessing"})
        )
        metric = MonaiMetricWrapper(
            monai_metric,
            pred_key=self.pred_key,
            label_key=self.label_key,
            optimum=self.optimum(),
            postprocessing=self.postprocessing,
        )
        return metric

    @field_validator("postprocessing", mode="after")
    @classmethod
    def _validate_postprocessing(
        cls, postprocessing: Union[Sequence[TransformOrConfig], PostprocessingHandler]
    ) -> PostprocessingHandler:
        """
        Puts postprocessing transforms in a PostprocessingHandler object.
        """
        if isinstance(postprocessing, list):
            return PostprocessingHandler(postprocessing)
        return postprocessing

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
