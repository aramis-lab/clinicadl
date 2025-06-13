import math
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config.base import (
    LossMetricConfig,
    MetricConfig,
)
from clinicadl.metrics.metrics import Metrics
from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class OneMetricEarlyStopping(Callback):
    """TO COMPLETE"""

    def __init__(
        self,
        metric: MetricConfig,
        patience: Optional[int] = None,
        min_delta: Optional[float] = 0.0,
        mode: Mode = Mode.MIN,
        check_finite: bool = True,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_finite = check_finite
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.check_bounds()
        self.is_better = self._get_comparison_function()
        self.reset()

    def _get_comparison_function(self):
        """Returns the comparison function."""
        if self.mode == Mode.MIN:
            return lambda value, best: value < best - self.min_delta
        elif self.mode == Mode.MAX:
            return lambda value, best: value > best + self.min_delta
        raise ValueError(f"Unknown mode: {self.mode}")

    def check_bounds(self):
        """TO COMPLETE"""
        if self.upper_bound is not None and self.lower_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("Upper bound should be greater than lower bound.")

    def reset(self) -> None:
        """Resets the epoch count and the best value."""
        if self.mode == Mode.MIN:
            self.best = np.inf
        elif self.mode == Mode.MAX:
            self.best = -np.inf
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.num_bad_epochs = 0

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        """
        Decides whether to stop the training or not, depending
        on the value of the last epoch.

        """
        df = config.metrics.df
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            value = df.loc[
                config.epoch, self.metric.name
            ]  # TODO: check if the df has the right columns and rows
            if value is None or pd.isna(value):
                raise ValueError(
                    f"Metric '{self.metric.name}' not found in DataFrame for epoch {config.epoch}."
                )
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

            return False

        raise ValueError("No df provided")


class EarlyStopping(Metrics, Callback):
    """
    TO COMPLETE
    """

    def __init__(
        self,
        metrics: list[
            Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]
        ],
        patience: Optional[Union[int, list[int]]] = None,
        min_delta: Optional[Union[float, list[float]]] = 0.0,
        mode: Union[Mode, list[Mode]] = Mode.MIN,
        check_finite: Union[bool, list[bool]] = True,
        upper_bound: Optional[Union[float, list[float]]] = None,
        lower_bound: Optional[Union[float, list[float]]] = None,
    ) -> None:
        self.metrics: list[MetricConfig] = self.check_metrics(metrics)
        len_metrics = len(metrics if isinstance(metrics, list) else [metrics])

        def check_list(value) -> list:
            if not isinstance(value, list):
                list_ = [value]
            else:
                list_ = value

            if len(list_) != 1 and len(list_) != len_metrics:
                raise ValueError(
                    f"List {list_} must have the same length as metrics: {len_metrics}"
                )
            elif len(list_) == 1:
                list_ = list_ * len_metrics
            return list_

        self.patience = check_list(patience)
        self.min_delta = check_list(min_delta)
        self.mode = check_list(mode)
        self.check_finite = check_list(check_finite)
        self.upper_bound = check_list(upper_bound)
        self.lower_bound = check_list(lower_bound)

        self.early_config_list = []

        for i, metric in enumerate(self.metrics):
            self.early_config_list.append(
                OneMetricEarlyStopping(
                    metric=metric,
                    patience=self.patience[i],
                    min_delta=self.min_delta[i],
                    mode=self.mode[i],
                    check_finite=self.check_finite[i],
                    upper_bound=self.upper_bound[i],
                    lower_bound=self.lower_bound[i],
                )
            )

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        for metric in self.early_config_list:
            metric.on_epoch_end(config=config, **kwargs)
