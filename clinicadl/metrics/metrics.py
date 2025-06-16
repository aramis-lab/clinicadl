from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch
from monai.metrics.confusion_matrix import ConfusionMatrixMetric
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.dictionary.words import EPOCH, LOSS_METRIC, METRICS
from clinicadl.losses.config import LossConfig
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
from clinicadl.utils.json import write_json


class Metrics(ABC):
    """
    Abstract base class for metrics.
    """

    @staticmethod
    def check_metrics(
        metrics: list[Union[MetricConfig, MonaiMetric, CustomMetric, LossConfig, Loss]],
    ) -> list[MetricConfig]:
        """TO COMPLETE"""

        metrics_config = []
        _confusion_metrics_name = []

        if not isinstance(metrics, list):
            metrics = [metrics]
        for metric in metrics:
            if isinstance(metric, MonaiMetric):
                config = get_metric_config(name=metric.__class__.__name__)
                metrics_config.append(config)

            elif isinstance(metric, LossConfig):
                metrics_config.append(
                    LossMetricConfig(
                        loss_fn=metric.get_object(), reduction=metric.reduction
                    )
                )

            elif isinstance(metric, MetricConfig) or isinstance(
                metric, LossMetricConfig
            ):
                metrics_config.append(metric)

            elif isinstance(metric, type(CustomMetric)):
                metrics_config.append(metric)

            elif isinstance(metric, type(Loss)):
                metrics_config.append(LossMetricConfig(loss_fn=metric))

        return metrics_config


class ClinicaDLMetrics(Metrics):
    """TO COMPLETE"""

    def __init__(
        self,
        metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ],
        loss: Loss,
    ):
        """
        Initialize the ClinicaDLMetrics instance.

        Parameters
        ----------
        metrics : MetricType
            Metric configuration or list of configurations.
        compute_train_metrics : bool
            Flag to compute training metrics.
        """
        if metrics is None:
            metrics = []  # TODO: which default metrics should we add?

        self.metrics = self.check_metrics(metrics)
        self._loss_metric = LossMetricConfig(loss_fn=loss)

        if self._loss_metric not in self.metrics:
            self.metrics.append(self._loss_metric)
            # TODO : check if 2 lossconifg, one for the loss and one as a metric, how to handle the name ? because a loss is a function and doesn't have a name

        self._callable_metrics = self.get_callable_metrics()
        self.df = self.init_df()

    def init_df(self) -> pd.DataFrame:
        """TO COMPLETE"""

        columns = [EPOCH]

        for metric in self._callable_metrics.values():
            if isinstance(metric, ConfusionMatrixMetric):
                for confusion_metric in metric.metric_name:
                    columns.append(confusion_metric)
            else:
                columns.append(metric.__class__.__name__)

        df = pd.DataFrame(columns=columns)
        df.set_index(EPOCH, inplace=True)
        return df

    def contains(
        self,
        metrics: list[Union[MetricConfig, MonaiMetric, CustomMetric, LossConfig, Loss]],
    ) -> bool:
        """TO COMPLETE"""
        metrics = self.check_metrics(metrics)
        return all(metric in self.metrics for metric in metrics)

    def add_metrics(
        self,
        metrics: list[
            Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]
        ],
    ) -> None:
        """
        Add metrics to the ClinicaDLMetrics instance.
        """
        self.metrics.extend(self.check_metrics(metrics))
        self._callable_metrics = self.get_callable_metrics()
        self.df = self.init_df()

    def get_callable_metrics(self) -> Dict[str, MonaiMetric]:
        """
        Retrieve the callable metrics.

        Returns
        -------
        Dict[str, MonaiMetric]
            Dictionary of callable metrics.
        """
        _callable_metrics: Dict[str, MonaiMetric] = {}

        for metric in self.metrics:
            if isinstance(metric, ConfusionMatrixMetricConfig):
                _callable_metrics[
                    metric.name
                ] = metric.get_object()  # TODO: for now same as others but need to check if several ConfusionMatrixMetricConfig are proposed with diofferent args
            elif isinstance(metric, LossMetricConfig):
                _callable_metrics[
                    metric.name
                ] = metric.get_object()  # TODO: handle loss name
            elif isinstance(metric, MetricConfig):
                _callable_metrics[metric.name] = metric.get_object()
            else:
                raise TypeError(
                    f"Unsupported metric type: {type(metric)}. Expected MetricConfig."
                )
        return _callable_metrics

    def _resetdf(self) -> None:
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
            self._resetdf()

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

    def get_loss(self) -> float:
        """
        Get the latest loss value.

        Returns
        -------
        float
            Aggregated loss.

        Raises
        ------
        ValueError
            If the loss metric is not present.
        """
        if LOSS_METRIC not in self._callable_metrics:
            raise ValueError("Loss not found in training metrics.")
        return self._callable_metrics[LOSS_METRIC].aggregate().item()

    def save(self, path: Path) -> None:
        """
        Persist the metrics to disk using selection metric file paths.

        Parameters
        ----------
        best_metrics : Dict[str, BestMetric]
            Mapping of metric names to their best-tracking wrappers.
        """
        self.df.to_csv(path, sep="\t", index=True)

    def to_dict(self) -> Dict[str, Optional[list[dict]]]:
        """
        Serialize the configuration to a dictionary.

        Returns
        -------
        dict
            Serialized metrics
        """
        return {
            METRICS: [metric.to_dict() for metric in self.metrics],
        }

    def write_json(self, json_path: Path) -> None:
        """
        Save the configuration to a JSON file.

        Parameters
        ----------
        json_path : Path
            Destination file path.
        """
        write_json(json_path, self.to_dict())
