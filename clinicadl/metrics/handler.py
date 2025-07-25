from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from pydantic import field_serializer, field_validator

from clinicadl.data.dataloader.batch import SimpleBatch
from clinicadl.dictionary.utils import SEP
from clinicadl.dictionary.words import (
    EPOCH,
    LOSS,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import MetricConfig, get_metric_config
from clinicadl.utils.config import ClinicaDLConfig

from .base import Metric
from .types import MetricOrConfig

CUSTOM_METRIC = "Custom metric passed by the user"


class _MetricProcessor(ClinicaDLConfig):
    """
    To check and convert metrics passed by the user.
    """

    metrics: dict[str, MetricOrConfig] = {}

    @field_validator("metrics", mode="after")
    @classmethod
    def _validate_names(
        cls, metrics: dict[str, MetricOrConfig]
    ) -> dict[str, MetricOrConfig]:
        """Checks that no metric is named 'loss'."""
        for name in metrics:
            if name == LOSS:
                raise ValueError(
                    "'loss' is a protected name, please choose another name for your metric."
                )
        return metrics

    @field_serializer("metrics")
    @classmethod
    def _serialize_metrics(
        cls, metrics: dict[str, MetricOrConfig]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of metrics that are not passed via
        MetricConfigs.
        """
        repr_ = []
        for metric in metrics:
            if isinstance(metric, MetricConfig):
                repr_.append(metric.to_dict())
            else:
                repr_.append(CUSTOM_METRIC + ": " + f"'{type(metric).__name__}'")

        return repr_

    def get_callable_metrics(self) -> dict[str, Metric]:
        """
        Gets the callable metrics.
        """
        callable_metrics: Dict[str, Metric] = {}

        for name, metric in self.metrics.items():
            if isinstance(metric, MetricConfig):
                callable_metrics[name] = metric.get_object()
            else:
                callable_metrics[name] = metric

        return callable_metrics

    def add_metrics(
        self,
        metrics: dict[str, MetricOrConfig],
    ) -> None:
        """
        Add metrics.
        """
        self.metrics = self.metrics | metrics


class MetricsHandler:
    """TO COMPLETE"""

    def __init__(
        self,
        loss: Optional[Loss] = None,
        metrics: Optional[dict[str, MetricOrConfig]] = None,
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
        if not metrics:
            metrics = {}

        self._metrics_processor = _MetricProcessor(metrics=metrics)
        if loss:
            self._metrics_processor.add_metrics({LOSS: loss})

        self.metrics = self._metrics_processor.metrics
        self._callable_metrics = self._metrics_processor.get_callable_metrics()

        self._df = self._init_df()
        self._detailed_df = self._init_detailed_df()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def detailed_df(self) -> pd.DataFrame:
        return self._detailed_df

    def _init_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with a column for each metric.
        """
        columns = list(self.metrics.keys())
        df = pd.DataFrame(columns=columns)

        return df

    def _init_detailed_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with a column for each metric,
        as well as columns "participant_id" and "session_id".
        """
        df = self._init_df()
        df.columns = df.columns.union([PARTICIPANT_ID, SESSION_ID])

        return df

    def add_metrics(
        self,
        metrics: dict[str, MetricOrConfig],
    ) -> None:
        """
        Add metrics to the MetricsHandler instance.
        """
        self._metrics_processor.add_metrics(metrics)
        self._callable_metrics = self._metrics_processor.get_callable_metrics()

        new_columns = list(metrics.keys())
        self._df = self._df.reindex(
            columns=self._df.columns.union(new_columns), fill_value=pd.NA
        )
        self._detailed_df = self._detailed_df.reindex(
            columns=self._detailed_df.columns.union(new_columns), fill_value=pd.NA
        )

    def reset(self, reset_df: bool = False) -> None:
        """
        Reset all metric states.

        Parameters
        ----------
        rest_df : bool
            If ``True``, also reset the metric DataFrame.
        """
        for metric in self._callable_metrics.values():
            metric.reset()
        if reset_df:
            self._df = self._init_df()
            self._detailed_df = self._init_detailed_df()

    def aggregate(self, epoch: Optional[int] = None) -> None:
        """
        Aggregate and store metric results.

        Parameters
        ----------
        epoch : int
            Current epoch.
        batch : Optional[int]
            Current batch (optional).
        """
        values = {}
        for name, metric in self._callable_metrics.items():
            values[name] = metric.aggregate()
        if epoch:
            values[EPOCH] = epoch

        new_df = pd.DataFrame([values])
        self._df = pd.concat([self._df, new_df], ignore_index=True)

    def __call__(self, batch: SimpleBatch, epoch: Optional[int] = None) -> None:
        """
        Update metrics using model predictions and ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y : torch.Tensor
            Ground truth labels.
        """
        participants = batch.get_field(PARTICIPANT)
        sessions = batch.get_field(SESSION)

        values = {}
        for name, metric in self._callable_metrics.items():
            values[name] = metric(batch)

        values = values | {PARTICIPANT_ID: participants, SESSION_ID: sessions}
        if epoch:
            values[EPOCH] = epoch
        new_df = pd.DataFrame(values)

        self._detailed_df = pd.concat([self._df, new_df], ignore_index=True)

    def save(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Persist the metrics to disk using selection metric file paths.

        Parameters
        ----------
        best_metrics : Dict[str, BestMetric]
            Mapping of metric names to their best-tracking wrappers.
        """
        self._df.to_csv(path, sep=SEP)
        if details_path:
            self._detailed_df.to_csv(details_path, sep=SEP)

    def write_json(self, json_path: Path) -> None:
        """
        Save the configuration to a JSON file.

        Parameters
        ----------
        json_path : Path
            Destination file path.
        """
        self._metrics_processor.write_json(json_path)

    @classmethod
    def from_json(
        cls, json_path: Path, loss: Optional[Loss] = None, **kwargs: Metric
    ) -> MetricsHandler:
        _dict = _MetricProcessor.read_json(json_path)
        for name, metric in _dict.items():
            if isinstance(metric, dict):
                _dict[name] = get_metric_config(**metric)
            else:
                if name in kwargs:
                    _dict[name] = kwargs[name]
                else:
                    raise ValueError(
                        f"Custom metric found for {name} in {str(json_path)}. "
                        "ClinicaDL can't read custom metric, so pass it to 'from_json' via "
                        f"{name}=<your-custom-metric>"
                    )

        metrics_processor = _MetricProcessor(metrics=_dict)

        return cls(loss=loss, metrics=metrics_processor.metrics)
