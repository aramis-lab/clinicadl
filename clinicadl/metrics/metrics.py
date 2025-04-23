from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.dictionary.words import EPOCH, LOSS_METRIC, METRICS, SELECTION_METRICS
from clinicadl.losses.types import Loss
from clinicadl.maps.split_dir.best_metric import BestMetric
from clinicadl.metrics.config import MetricConfig, get_metric_config
from clinicadl.metrics.config.base import LossMetricConfig
from clinicadl.utils.json import read_json, write_json

MetricType = Union[MetricConfig, list[MetricConfig]]


class ClinicaDLMetrics:
    """
    Handles the configuration, computation, and aggregation of training and evaluation metrics
    for deep learning models within the ClinicaDL framework.
    """

    def __init__(
        self,
        metrics: MetricType,
        selection_metrics: Optional[MetricType] = None,
        compute_train_metrics: bool = False,
    ):
        """
        Initialize ClinicaDLMetrics.

        Parameters
        ----------
        metrics : MetricType
            A MetricConfig or a list of MetricConfig to evaluate.
        selection_metrics : Optional[MetricType]
            Subset of `metrics` used to select best-performing models.
        compute_train_metrics : bool (default False)
            Whether to compute metrics during training.
        """
        self.metrics: Dict[str, MetricConfig] = (
            {metrics.name: metrics}
            if isinstance(metrics, MetricConfig)
            else {metric.name: metric for metric in metrics}
        )
        self.callable_metrics: Dict[str, MonaiMetric] = {
            metric.name: metric.get_object() for metric in self.metrics.values()
        }

        self.selection_metrics: Optional[Dict[str, MetricConfig]] = None
        if selection_metrics:
            self.selection_metrics = (
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
        self.reset_df()

    def reset_df(self) -> None:
        """
        Initialize or reset the internal DataFrame for storing aggregated metric values.
        """
        self._df = pd.DataFrame(
            columns=[EPOCH] + [metric.name for metric in self.metrics.values()]
        )
        self._df.set_index(EPOCH, inplace=True)

    @classmethod
    def from_json(cls, json_path: Path) -> ClinicaDLMetrics:
        """
        Create an instance from a JSON configuration file.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file.

        Returns
        -------
        ClinicaDLMetrics
            Configured instance.
        """
        metrics_dict = read_json(json_path)
        return cls.from_dict(metrics_dict)

    @classmethod
    def from_dict(cls, metrics_dict: dict) -> ClinicaDLMetrics:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary with metric and selection_metric configuration.

        Returns
        -------
        ClinicaDLMetrics
            Configured instance.
        """
        metrics = [get_metric_config(**m) for m in metrics_dict[METRICS]]
        selection_metrics = (
            [get_metric_config(**m) for m in metrics_dict[SELECTION_METRICS]]
            if metrics_dict[SELECTION_METRICS]
            else None
        )

        return cls(metrics=metrics, selection_metrics=selection_metrics)

    def to_dict(self) -> Dict[str, Optional[list[dict]]]:
        """
        Serialize the configuration to a dictionary.

        Returns
        -------
        dict
            Serialized metrics and selection_metrics.
        """
        return {
            METRICS: [metric.model_dump() for metric in self.metrics.values()],
            SELECTION_METRICS: (
                [metric.model_dump() for metric in self.selection_metrics.values()]
                if self.selection_metrics
                else None
            ),
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

    def _init_with_loss(self, loss: Loss) -> None:
        """
        Extend metrics with a loss metric.

        Parameters
        ----------
        loss : Loss
            Loss object to be added and tracked.
        """
        loss_metric_config = LossMetricConfig(loss_fn=loss)
        self.metrics[LOSS_METRIC] = loss_metric_config
        self.callable_metrics[LOSS_METRIC] = loss_metric_config.get_object()

        if self.selection_metrics is None:
            self.selection_metrics = {LOSS_METRIC: loss_metric_config}

    def reset(self, df: bool = False) -> None:
        """
        Reset all metric states.

        Parameters
        ----------
        df : bool
            If True, also reset the DataFrame.
        """
        for metric in self.callable_metrics.values():
            metric.reset()

        if df:
            self.reset_df()

    def aggregate(self, epoch: int, batch: Optional[int] = None) -> None:
        """
        Aggregate and store metric results.

        Parameters
        ----------
        epoch : int
            Current epoch.
        batch : Optional[int]
            Current batch (optional).
        """
        for name, metric in self.callable_metrics.items():
            value = metric.aggregate().item()
            if batch is not None:
                self._df.at[(epoch, batch), name] = value
            else:
                self._df.at[epoch, name] = value

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update metrics using model predictions and ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y : torch.Tensor
            Ground truth labels.
        """
        for metric in self.callable_metrics.values():
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
        if LOSS_METRIC not in self.callable_metrics:
            raise ValueError("Loss not found in training metrics.")
        return self.callable_metrics[LOSS_METRIC].aggregate().item()

    def get_value(self, epoch: int, name: str) -> float:
        """
        Retrieve a metric value from the DataFrame.

        Parameters
        ----------
        epoch : int
            Epoch to retrieve.
        name : str
            Metric name.

        Returns
        -------
        float
            Metric value, or NaN if not found.

        Raises
        ------
        KeyError
            If metric name is invalid.
        """
        if name not in self._df.columns:
            raise KeyError(
                f"Metric '{name}' not found. Available: {self._df.columns.tolist()}"
            )
        if epoch not in self._df.index:
            return np.nan
        return self._df.at[epoch, name]

    def save(self, best_metrics: Dict[str, BestMetric]) -> None:
        """
        Persist the metrics to disk using selection metric file paths.

        Parameters
        ----------
        best_metrics : Dict[str, BestMetric]
            Mapping of metric names to their best-tracking wrappers.
        """
        if not self.selection_metrics:
            raise RuntimeError("Cannot save metrics without selection_metrics.")
        for name in self.selection_metrics:
            self._df.to_csv(best_metrics[name].val.metrics_tsv, sep="\t", index=True)
