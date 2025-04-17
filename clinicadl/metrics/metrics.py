from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.losses.types import Loss
from clinicadl.metrics import ImplementedMetric
from clinicadl.metrics.config import MetricConfig, get_metric_config
from clinicadl.metrics.config.base import LossMetricConfig
from clinicadl.utils.json import read_json, write_json

MetricType = Union[MetricConfig, list[MetricConfig]]
LOSS = "Loss"


class ClinicaDLMetrics:
    """TO COMPLETE"""

    def __init__(
        self,
        metrics: MetricType,
        selection_metrics: Optional[MetricType] = None,
        compute_train_metrics: bool = False,
    ):
        self.metrics: Dict[str, MetricConfig] = (
            {metrics.name: metrics}
            if isinstance(metrics, MetricConfig)
            else {metric.name: metric for metric in metrics}
        )
        self.callable_metrics: Dict[str, MonaiMetric] = {
            metric.name: metric.get_object() for metric in self.metrics.values()
        }

        if selection_metrics:
            self.selection_metrics: Dict[str, MetricConfig] = (
                {selection_metrics.name: selection_metrics}
                if isinstance(selection_metrics, MetricConfig)
                else {metric.name: metric for metric in selection_metrics}
            )

        if self.selection_metrics and not set(self.selection_metrics).issubset(
            set(self.metrics)
        ):
            raise ValueError(
                f"Selection metrics ({self.selection_metrics}) must be one of the provided metrics ({self.metrics})."
            )

        self.compute_train_metrics = compute_train_metrics

        self._df = pd.DataFrame(
            columns=["epoch"] + [metric.name for metric in self.metrics.values()]
        )
        self._df.set_index("epoch", inplace=True)

    @classmethod
    def from_json(cls, json_path: Path) -> ClinicaDLMetrics:
        """
        Create a ClinicaDLMetrics instance from a JSON file.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file.

        Returns
        -------
        ClinicaDLMetrics
            An instance of the ClinicaDLMetrics class.
        """

        metrics_dict = read_json(json_path)

        return cls.from_dict(metrics_dict)

    @classmethod
    def from_dict(cls, metrics_dict: dict) -> ClinicaDLMetrics:
        """
        Create a ClinicaDLMetrics instance from a dictionary.

        Parameters
        ----------
        metrics : dict
            Dictionary containing the metrics configuration.

        Returns
        -------
        ClinicaDLMetrics
            An instance of the ClinicaDLMetrics class.
        """
        metrics = []
        for metric_dict in metrics_dict["metrics"]:
            metric_config = get_metric_config(**metric_dict)
            metrics.append(metric_config)

        if metrics_dict["selection_metrics"]:
            selection_metrics = []
            for selection_metric_dict in metrics_dict["selection_metrics"]:
                selection_metric_config = get_metric_config(**selection_metric_dict)
                selection_metrics.append(selection_metric_config)
        else:
            selection_metrics = None

        return cls(metrics=metrics, selection_metrics=selection_metrics)

    def to_dict(self) -> dict:
        """
        Convert the metrics configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary containing the metrics configuration.
        """
        metrics_dict = {
            "metrics": [metric.model_dump() for metric in self.metrics.values()],
            "selection_metrics": [
                metric.model_dump() for metric in self.selection_metrics.values()
            ]
            if self.selection_metrics
            else None,
        }
        return metrics_dict

    def write_json(self, json_path: Path):
        """
        Save the metrics configuration to a JSON file.
        """
        write_json(json_path, self.to_dict())

    def _init_with_loss(self, loss: Loss):
        """
        Initialize the DataFrame to store training loss and metrics.
        """
        loss_metric_config = LossMetricConfig(loss_fn=loss)

        self.metrics[LOSS] = loss_metric_config
        self.callable_metrics[LOSS] = loss_metric_config.get_object()

        if self.selection_metrics is None:
            self.selection_metrics = {LOSS: loss_metric_config}

    def reset(self):
        """
        Reset the metrics to their initial state.
        """
        for metric in self.callable_metrics.values():
            metric.reset()

    def aggregate(self, epoch: int, batch: Optional[int] = None):
        """
        Aggregate the metrics across all batches.
        """
        for name, metric in self.callable_metrics.items():
            if batch:
                self._df.at[(epoch, batch), name] = metric.aggregate().item()
            else:
                self._df.at[epoch, name] = metric.aggregate().item()

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Update the training metrics with predictions and ground truth.
        """
        for metric in self.callable_metrics.values():
            metric(y_pred, y)

    def get_loss(self) -> float:
        """
        Get the loss value from the training metrics.
        """
        if LOSS not in self.callable_metrics:
            raise ValueError("Loss not found in training metrics.")
        return self.callable_metrics[LOSS].aggregate().get_item()
