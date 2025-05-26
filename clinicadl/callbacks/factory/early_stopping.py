import math
from enum import Enum
from typing import Optional, Union

import numpy as np
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.metrics import LossMetricConfig, MetricConfig
from clinicadl.utils.config import ClinicaDLConfig

from .base import Callback


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class EarlyStopping(Callback):
    def __init__(
        self,
        metrics: list[Union[MetricConfig, MonaiMetric, LossConfig, Loss]],
        patience: Optional[Union[int, list[int]]] = None,
        min_delta: Optional[Union[float, list[float]]] = 0.0,
        mode: Union[Mode, list[Mode]] = Mode.MIN,
        check_finite: Union[bool, list[bool]] = True,
        upper_bound: Optional[Union[float, list[float]]] = None,
        lower_bound: Optional[Union[float, list[float]]] = None,
    ) -> None:
        self.metrics = self.check_metrics(metrics)
        len_metrics = len(metrics if isinstance(metrics, list) else [metrics])

        def check_list(value):
            if value:
                if not isinstance(value, list):
                    value = [value]
                return value

        def check_len(list_: Optional[list]):
            if list_:
                if len(list_) != 1 and len(list_) != len_metrics:
                    raise ValueError(
                        f"List {list_} must have the same length as metrics: {len_metrics}"
                    )
                elif len(list_) == 1:
                    list_ = list_ * len_metrics
            return list_

        self.patience = check_len(check_list(patience))
        self.min_delta = check_len(check_list(min_delta))
        self.mode = check_len(check_list(mode))
        self.check_finite = check_len(check_list(check_finite))
        self.upper_bound = check_len(check_list(upper_bound))
        self.lower_bound = check_len(check_list(lower_bound))

        self.check_bounds()
        self.is_better = self._get_comparison_function()
        self.reset()

    def check_metrics(
        self, metrics: list[Union[MetricConfig, MonaiMetric, LossConfig, Loss]]
    ):
        if not isinstance(metrics, list):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            if isinstance(metric, LossConfig):
                metrics[i] = LossMetricConfig(loss_fn=metric.get_object())
            elif isinstance(metric, Loss):
                metrics[i] = LossMetricConfig(loss_fn=metric)
        return metrics

    def check_bounds(self):
        """TO COMPLETE"""
        if self.upper_bound is not None and self.lower_bound is not None:
            for i, low in enumerate(self.lower_bound):
                if low < self.upper_bound[i]:
                    raise ValueError("Upper bound should be greater than lower bound.")

    def _get_comparison_function(self):
        """Returns the comparison function."""
        if self.mode == Mode.MIN:
            return lambda value, best: value < best - self.min_delta
        if self.mode == Mode.MAX:
            return lambda value, best: value > best + self.min_delta

    def reset(self) -> None:
        """Resets the epoch count and the best value."""
        if self.mode == Mode.MIN:
            self.best = np.inf
        if self.mode == Mode.MAX:
            self.best = -np.inf
        self.num_bad_epochs = 0

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        """
        Decides whether to stop the training or not, depending
        on the value of the last epoch.

        Parameters
        ----------
        value : float
            The value obtained during the last epoch.

        Returns
        -------
        bool
            The decision.
        """
        value = kwargs.get("value", None)
        if value:
            if self.check_finite and (math.isinf(value) or math.isnan(value)):
                return True

            if self.upper_bound is not None and (value > self.upper_bound):
                return True

            if self.lower_bound is not None and (value < self.lower_bound):
                return True

            if self.patience is None:
                return False

            if self.is_better(value, self.best):
                self.num_bad_epochs = 0
                self.best = value
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                return True
        raise ValueError("No value provided")

    def on_batch_begin(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, **kwargs):
        pass

    def on_backward_begin(self, **kwargs):
        pass

    def on_validation_begin(self, **kwargs):
        pass

    def on_validation_end(self, **kwargs):
        pass
