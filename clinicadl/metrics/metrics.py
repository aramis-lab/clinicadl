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
        Initialize the ClinicaDLMetrics instance.

        Parameters
        ----------
        metrics : MetricType
            Metric configuration or list of configurations.
        selection_metrics : Optional[MetricType]
            Optional selection metrics configuration.
        compute_train_metrics : bool
            Flag to compute training metrics.
        """
        self.metrics: Dict[str, MetricConfig] = self._check_list_metrics(metrics)
        self.selection_metrics: Optional[Dict[str, MetricConfig]] = (
            self._check_list_metrics(selection_metrics) if selection_metrics else None
        )

        self.compute_train_metrics = compute_train_metrics
        self._df = pd.DataFrame(
            columns=[EPOCH] + [metric.name for metric in self.metrics.values()]
        )
        self._df.set_index(EPOCH, inplace=True)
        self._callable_metrics: Dict[str, MonaiMetric] = {
            name: metric.get_object() for name, metric in self.metrics.items()
        }

    def _check_list_metrics(self, v: MetricType) -> Dict[str, MetricConfig]:
        """
        Validate that the input is a list of MetricConfig or a single MetricConfig.

        Parameters
        ----------
        v : Union[MetricConfig, list[MetricConfig]]
            Input metric configuration.

        Returns
        -------
        Dict[str, MetricConfig]
            Dictionary of validated metrics.
        """
        if isinstance(v, MetricConfig):
            return {v.name: v}
        elif isinstance(v, list):
            metrics = {}
            for metric in v:
                if not isinstance(metric, MetricConfig):
                    raise TypeError(f"Expected MetricConfig, got {type(metric)}")
                if metric.name in metrics:
                    raise ValueError(f"Duplicate metric name '{metric.name}' found.")
                metrics[metric.name] = metric
            return metrics
        else:
            raise TypeError(
                f"Expected MetricConfig or list of MetricConfig, got {type(v)}"
            )

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
        metrics = [get_metric_config(**m) for m in metrics_dict.get(METRICS, [])]
        selection_metrics = (
            [get_metric_config(**m) for m in metrics_dict.get(SELECTION_METRICS, [])]
            if selection_metrics in metrics_dict
            else None
        )
        compute_train_metrics = metrics_dict.get("compute_train_metrics", False)

        return cls(
            metrics=metrics,
            selection_metrics=selection_metrics,
            compute_train_metrics=compute_train_metrics,
        )

    def _reset_df(self) -> None:
        """
        Initialize or reset the internal DataFrame for storing aggregated metric values.
        """
        self._df.drop(self._df.index, inplace=True)

    def _configure_loss_tracking(self, loss: Loss) -> None:
        """
        Configure internal metric tracking to include the loss function.

        This method performs the following:
        - Registers the loss function as a metric (`LOSS_METRIC`) in both `metrics` and, if necessary, `selection_metrics`.
        - Initializes or updates the callable versions of all metrics.
        - Resets the internal tracking dataframe to include the loss.

        Parameters
        ----------
        loss : Loss
            Loss object to be tracked alongside other evaluation metrics.
        """

        loss_metric_config = LossMetricConfig(loss_fn=loss)
        self.metrics[LOSS_METRIC] = loss_metric_config
        self._callable_metrics[LOSS_METRIC] = loss_metric_config.get_object()

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
        if self._callable_metrics is None:
            raise RuntimeError(
                "Metrics not initialized. Call _configure_loss_tracking first."
            )

        for metric in self._callable_metrics.values():
            metric.reset()

        if df:
            self._reset_df()

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
        for name, metric in self._callable_metrics.items():
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
