from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from pydantic import field_serializer

from clinicadl.data.dataloader import Batch
from clinicadl.dictionary.utils import SEP
from clinicadl.dictionary.words import (
    EPOCH,
    LOSS,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.metrics.config import LossMetricConfig, MetricConfig, get_metric_config
from clinicadl.utils.config import ClinicaDLConfig

from .base import Metric
from .types import MetricOrConfig

CUSTOM_METRIC = "Custom metric passed by the user"


class _MetricProcessor(ClinicaDLConfig):
    """
    To check and convert metrics passed by the user.
    """

    metrics: dict[str, MetricOrConfig] = {}

    @field_serializer("metrics")
    @classmethod
    def _serialize_metrics(
        cls, metrics: dict[str, MetricOrConfig]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of metrics that are not passed via
        MetricConfigs.
        """
        repr_ = {}
        for name, metric in metrics.items():
            if isinstance(metric, MetricConfig):
                repr_[name] = metric.to_dict()
            else:
                repr_[name] = CUSTOM_METRIC + ": " + f"'{type(metric).__name__}'"

        return repr_

    @classmethod
    def from_json(cls, json_path: Path, **kwargs) -> _MetricProcessor:
        """
        Reads the serialized config class from a JSON file.
        """
        dict_: dict = cls.read_json(json_path)["metrics"]
        for name, metric in dict_.items():
            if isinstance(metric, dict):
                dict_[name] = get_metric_config(**metric)
            else:
                if name in kwargs:
                    dict_[name] = kwargs[name]
                else:
                    raise ValueError(
                        f"Custom metric found for '{name}' in {str(json_path)}. "
                        "ClinicaDL can't read custom metric, so pass it to 'from_json' via "
                        f"{name}=<your-custom-metric>"
                    )

        return _MetricProcessor(metrics=dict_)

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
        Adds metrics.
        """
        for name in metrics:
            if name in self.metrics:
                raise ValueError(f"A metric named '{name}' already exists!")
        self.metrics = self.metrics | metrics


class MetricsHandler:
    """
    To handle the metrics during a validation phase.

    This object accepts as inputs raw metrics (i.e. objects that inherits from
    :py:class:`clinicadl.metrics.Metric`) or config classes. MetricsHandler will
    convert config classes to obtain the associated callable.

    MetricsHandler is itself a callable that works like :py:class:`monai.metricsCumulativeIterationMetric`,
    with :py:meth:`reset` and :py:meth:`aggregate` methods. So, it can be used like a :py:class:`clinicadl.metrics.Metric`
    object.

    The results are stored in DataFrames (:py:attr:`df` and :py:attr:`detailed_df`), that can be saved with
    :py:meth:`save`.

    Parameters
    ----------
    loss : Optional[LossMetricConfig], default=None
        A loss to add to the metrics. It must be passed via
        :py:class:`clinicadl.metrics.config.LossMetricConfig`.
    **metrics : MetricConfig
        Metrics to add to the MetricsHandler. They must be passed as
        :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
    """

    def __init__(
        self,
        loss: Optional[LossMetricConfig] = None,
        **metrics: MetricOrConfig,
    ):
        if not metrics:
            metrics = {}

        self._metrics_processor = _MetricProcessor(metrics=metrics)
        self.metrics = deepcopy(self._metrics_processor.metrics)
        self._callable_metrics = self._metrics_processor.get_callable_metrics()

        if loss:
            self._add_loss(loss)

        self._df = self._init_df()
        self._detailed_df = self._init_detailed_df()

    @property
    def df(self) -> pd.DataFrame:
        """
        The :py:class:`pandas.DataFrame` containing the aggregated results, i.e. the results on
        the whole dataset obtained by calling :py:meth:`aggregate`.
        """
        return self._df

    @property
    def detailed_df(self) -> pd.DataFrame:
        """
        The :py:class:`pandas.DataFrame` containing the detailed results,
        i.e. the results for each image.
        """
        return self._detailed_df

    def _init_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with a column for each metric.
        """
        return pd.DataFrame(columns=list(self.metrics.keys()))

    def _init_detailed_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with a column for each metric,
        as well as columns "participant_id" and "session_id".
        """
        columns = list(self.metrics.keys()) + [PARTICIPANT_ID, SESSION_ID]

        return pd.DataFrame(columns=columns)

    def add_metrics(
        self,
        loss: Optional[LossMetricConfig] = None,
        **metrics: MetricOrConfig,
    ) -> None:
        """
        Add metrics to the MetricsHandler instance.

        Parameters
        ----------
        loss : Optional[LossMetricConfig], default=None
            A loss to add to the metrics. It must be passed via
            :py:class:`clinicadl.metrics.config.LossMetricConfig`.
        **metrics : MetricConfig
            Metrics to add to the MetricsHandler. They must be passed as
            :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
        """
        self._metrics_processor.add_metrics(metrics)
        self.metrics = self._metrics_processor.metrics
        self._callable_metrics = self._metrics_processor.get_callable_metrics()

        if loss:
            self._add_loss(loss=loss)

        self._df = self._df.reindex(
            columns=self._df.columns.union(self.metrics.keys()), fill_value=pd.NA
        )
        self._detailed_df = self._detailed_df.reindex(
            columns=self._detailed_df.columns.union(self.metrics.keys()),
            fill_value=pd.NA,
        )

    def reset(self, reset_df: bool = False) -> None:
        """
        Reset all metric states.

        Parameters
        ----------
        reset_df : bool, default=False
            If ``True``, also reset the DataFrames containing the results.

        See Also
        --------
        :py:meth:`monai.metrics.Cumulative.reset`
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
        epoch : Optional[int], default=None
            Current epoch. This information will be added in the DataFrame.

        See Also
        --------
        :py:meth:`monai.metrics.Cumulative.aggregate`
        """
        values = {}
        for name, metric in self._callable_metrics.items():
            values[name] = metric.aggregate()
        if epoch is not None:
            values[EPOCH] = epoch

        new_df = pd.DataFrame([values])
        self._df = pd.concat([self._df, new_df], ignore_index=True)

        if epoch is not None:
            try:
                self._df = self._df.astype({EPOCH: int})
            except pd.errors.IntCastingNaNError:
                pass

    def __call__(self, batch: Batch, epoch: Optional[int] = None) -> None:
        """
        Updates metrics with a new batch.

        Parameters
        ----------
        batch : Batch
            The batch, with the predictions, and the ground truths if required
            by some metrics.
        epoch : Optional[int], default=None
            Current epoch. This information will be added in the DataFrame.
        """
        participants = batch.get_field(PARTICIPANT)
        sessions = batch.get_field(SESSION)

        values = {}
        for name, metric in self._callable_metrics.items():
            values[name] = metric(batch)

        values = values | {PARTICIPANT_ID: participants, SESSION_ID: sessions}
        if epoch is not None:
            values[EPOCH] = epoch
        new_df = pd.DataFrame(values)

        self._detailed_df = pd.concat([self._detailed_df, new_df], ignore_index=True)

        if epoch is not None:
            try:
                self._detailed_df = self._detailed_df.astype({EPOCH: int})
            except pd.errors.IntCastingNaNError:
                pass

    def save(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Saves the DataFrames containing the results.

        Parameters
        ----------
        path : Path
            The path for the DataFrame with the aggregated results.
        details_path: Optional[Path], default=None
            The path for the DataFrame with the detailed results.
            If ``None``, this DataFrame will not be saved.
        """
        self._df.to_csv(path, sep=SEP, index=False)
        if details_path:
            self._detailed_df.to_csv(details_path, sep=SEP, index=False)

    def write_json(self, json_path: Path) -> None:
        """
        Save the configuration to a JSON file.

        .. note::
            The loss metric is not saved in this file.

        Parameters
        ----------
        json_path : Path
            Destination file path.
        """
        self._metrics_processor.write_json(json_path)

    @classmethod
    def from_json(
        cls,
        json_path: Path,
        loss: Optional[LossMetricConfig] = None,
        **metrics: MetricConfig,
    ) -> MetricsHandler:
        """
        Creates a MetricsHandler from a JSON file saved with :py:meth:`write_json`.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file.
        loss : Optional[LossMetricConfig], default=None
            A loss to add to the metrics. Indeed, the loss is not save in the JSON file by
            :py:meth:`write_json`.
        **metrics : MetricConfig
            Other metrics to add in the MetricsHandler. It is also a way to pass a custom
            metric that otherwise cannot be read in the JSON file.
        """
        metrics_processor = _MetricProcessor.from_json(json_path, **metrics)
        return cls(loss=loss, **metrics_processor.metrics)

    def _add_loss(self, loss: LossMetricConfig) -> None:
        """
        To add the loss to the metrics.
        """
        if LOSS in self.metrics:
            raise ValueError("You already passed a loss!")
        if not isinstance(loss, LossMetricConfig):
            raise ValueError(
                f"Loss must be passed as a LossMetricConfig. Got {type(loss).__name__}"
            )
        self.metrics[LOSS] = loss
        self._callable_metrics[LOSS] = loss.get_object()
