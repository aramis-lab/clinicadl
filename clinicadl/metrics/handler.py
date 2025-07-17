from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch
from monai.metrics.confusion_matrix import ConfusionMatrixMetric
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.dictionary.words import EPOCH, LOSS_METRIC, METRICS
from clinicadl.losses.config import LossConfig, get_loss_function_config
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import (
    ConfusionMatrixMetricConfig,
    CustomMetric,
    MetricConfig,
)
from clinicadl.metrics.config.base import (
    LossMetricConfig,
    MetricConfig,
)
from clinicadl.metrics.config.factory import get_metric_config
from clinicadl.utils.json import read_json, write_json

from .types import MetricType


class Metrics(ABC):
    """
    Abstract base class for metrics.
    """

    @staticmethod
    def check_metrics(
        metrics: Optional[dict[str, MetricType]],
    ) -> dict[str, MetricConfig]:
        """TO COMPLETE"""

        metrics_config: dict[str, MetricConfig] = {}

        if metrics is not None:
            if not isinstance(metrics, dict):
                raise TypeError(
                    f"Metrics must be a dictionary, got {type(metrics)} instead."
                )

            for metric_name, metric in metrics.items():
                if isinstance(metric, MonaiMetric):
                    config = get_metric_config(
                        name=metric.__class__.__name__, **metric.__dict__
                    )  # TODO : check if it works when doing unittests
                    metrics_config[metric_name] = config

                elif isinstance(metric, LossConfig):
                    metrics_config[metric_name] = LossMetricConfig(
                        loss_fn=metric.get_object(), reduction=metric.reduction
                    )

                elif isinstance(metric, MetricConfig) or isinstance(
                    metric, LossMetricConfig
                ):
                    metrics_config[metric_name] = metric

                elif isinstance(metric, type(CustomMetric)):
                    metrics_config[metric_name] = metric

                elif isinstance(metric, Loss):
                    metrics_config[metric_name] = LossMetricConfig(loss_fn=metric)

        return metrics_config


class MetricsHandler(Metrics):
    """TO COMPLETE"""

    def __init__(
        self,
        loss: Loss,
        metrics: Optional[dict[str, MetricType]] = None,
    ):
        """
        Initialize the MetricsHandler instance.

        Parameters
        ----------
        metrics : MetricType
            Metric configuration or list of configurations.
        compute_train_metrics : bool
            Flag to compute training metrics.
        """

        self.metrics = self.check_metrics(metrics=metrics)
        self._loss = loss
        self._loss_metric = LossMetricConfig(loss_fn=loss)
        if "loss" not in self.metrics:
            self.metrics["loss"] = self._loss_metric
            # TODO : check if 2 lossconifg, one for the loss and one as a metric, how to handle the name ? because a loss is a function and doesn't have a name

        self._callable_metrics = self.get_callable_metrics()
        self.df = self._init_df()

    def _init_df(self) -> pd.DataFrame:
        """TO COMPLETE"""

        columns = [EPOCH]

        for name, metric in self._callable_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                for confusion_metric in metric.metric_name:
                    columns.append(confusion_metric)
            else:
                columns.append(name)

        df = pd.DataFrame(columns=columns)
        df.set_index(EPOCH, inplace=True)
        return df

    def add_metrics(
        self,
        metrics: dict[str, MetricType],
    ) -> None:
        """
        Add metrics to the MetricsHandler instance.
        """
        add_metric = self.check_metrics(metrics)
        new_callable_metrics = self.get_callable_metrics()
        for name, _callable in new_callable_metrics.items():
            if name not in self._callable_metrics.keys():
                self._callable_metrics[name] = _callable
                self.metrics[name] = add_metric[name]

        new_df = self._init_df()
        self.df = self.df.reindex(
            columns=self.df.columns.union(new_df.columns), fill_value=pd.NA
        )

    def get_callable_metrics(self) -> Dict[str, MonaiMetric]:
        """
        Retrieve the callable metrics.

        Returns
        -------
        Dict[str, MonaiMetric]
            Dictionary of callable metrics.
        """
        _callable_metrics: Dict[str, MonaiMetric] = {}

        for name, config in self.metrics.items():
            try:
                _callable_metrics[name] = config.get_object()
            except TypeError:
                raise TypeError(
                    f"The provided metric {name} doesn't have a get_object method."
                )

        return _callable_metrics

    def _reset_df(self) -> None:
        """
        Initialize or reset the internal DataFrame for storing aggregated metric values.
        """
        self.df.drop(self.df.index, inplace=True)

    def reset(self, df: bool = False) -> None:
        """
        Reset all metric states.

        Parameters
        ----------
        df : bool
            If True, also reset the DataFrame.
        """
        for metric in self._callable_metrics.values():
            metric.reset()
        if df:
            self._reset_df()

    def aggregate(self, epoch: int) -> None:
        """
        Aggregate and store metric results.

        Parameters
        ----------
        epoch : int
            Current epoch.
        batch : Optional[int]
            Current batch (optional).
        """
        for name, metric in self._callable_metrics.items():
            value = metric.aggregate()
            if isinstance(metric, ConfusionMatrixMetric):
                for i, _name in enumerate(metric.metric_name):
                    self.df.at[epoch, _name] = value[i].item()
            else:
                self.df.at[epoch, name] = value.item()

    def __call__(
        self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        """
        Update metrics using model predictions and ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y : torch.Tensor
            Ground truth labels.
        """
        for metric in self._callable_metrics.values():
            metric(y_pred, y)

    def save(self, path: Path) -> None:
        """
        Persist the metrics to disk using selection metric file paths.

        Parameters
        ----------
        best_metrics : Dict[str, BestMetric]
            Mapping of metric names to their best-tracking wrappers.
        """
        self.df.to_csv(path, sep="\t", index=True)

    def write_json(self, json_path: Path) -> None:
        """
        Save the configuration to a JSON file.

        Parameters
        ----------
        json_path : Path
            Destination file path.
        """
        json_dict = {name: metric.to_dict() for name, metric in self.metrics.items()}
        write_json(json_path, json_dict)

    @classmethod
    def from_json(cls, json_path: Path) -> dict[str, MetricConfig]:
        json_path = Path(json_path)
        _dict = read_json(json_path=json_path)

        for name, metric in _dict.items():
            if "loss_fn" in metric:
                metric["loss_fn"] = get_loss_function_config(
                    name=metric["loss_fn"]["name"], **metric["loss_fn"]["params"]
                ).get_object()
            _dict[name] = get_metric_config(**metric)
        return _dict
