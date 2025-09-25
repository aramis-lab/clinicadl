from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING

from monai.metrics import LossMetric

from clinicadl.losses.types import Loss
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.factories import get_defaults_from

from ..base import Metric
from ..enum import Optimum
from ..monai_wrapper import MonaiMetricWrapper
from .base import MetricConfig, _GetNotNansConfig

if TYPE_CHECKING:
    from clinicadl.models import ClinicaDLModel

logger = getLogger("clinicadl.metrics.loss")

LOSS_METRIC_MONAI_DEFAULTS = get_defaults_from(LossMetric)


class LossMetricConfig(MetricConfig, _GetNotNansConfig):
    """Special config class to use a loss function as a metric."""

    loss_name: str = "loss"

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN

    def get_object(self, model: ClinicaDLModel) -> Metric:
        """
        Returns the metric associated to this configuration,
        parametrized with the parameters passed by the user.

        Parameters
        ----------
        model : ClinicaDLModel
            The :py:class:`clinicadl.model.ClinicaDLModel` where the loss is.

        Returns
        -------
        Metric:
            The associated metric.
        """
        losses = model.get_loss_functions()
        try:
            loss = losses[self.loss_name]
        except KeyError as exc:
            raise ClinicaDLArgumentError(
                f"In LossMetricConfig, loss_name='{self.loss_name}' but there is no such loss (returned by the 'get_loss_functions' method of you ClinicaDLModel). "
                f"Losses are: {list(losses.keys())}"
            ) from exc

        loss, reduction = self._check_reduction(loss)

        monai_metric = LossMetric(
            loss_fn=loss, reduction=reduction, get_not_nans=self.get_not_nans
        )
        metric = MonaiMetricWrapper(
            monai_metric,
            pred_key=self.pred_key,
            label_key=self.label_key,
            optimum=self.optimum(),
            postprocessing=self.postprocessing,
        )
        return metric

    def _check_reduction(self, loss: Loss) -> tuple[Loss, str]:
        """Remove the reduction of the loss to put it at the metric level."""
        try:
            loss_reduction = getattr(loss, "reduction")
        except AttributeError as exc:
            raise ClinicaDLArgumentError(
                f"The loss '{self.loss_name}' (returned by the 'get_loss_functions' method of you ClinicaDLModel) "
                "doesn't have a 'reduction' attribute, so ClinicaDL can't compute the validation loss at the image level."
            ) from exc

        loss = deepcopy(loss)
        setattr(loss, "reduction", "none")

        return loss, loss_reduction
