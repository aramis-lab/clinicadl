from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import pandas as pd
from pydantic import field_serializer

from clinicadl.data.dataloader import Batch
from clinicadl.dictionary.utils import SEP
from clinicadl.dictionary.words import (
    EPOCH,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.metrics.config import LossMetricConfig, MetricConfig, get_metric_config
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError

from .base import Metric
from .types import MetricOrConfig

if TYPE_CHECKING:
    from clinicadl.models import ClinicaDLModel

CUSTOM_METRIC = "Custom metric passed by the user"


class MetricsHandlerConfig(ClinicaDLConfig):
    """
    To check and convert metrics passed by the user.
    """

    metrics: dict[str, MetricOrConfig] = {}

    @field_serializer("metrics")
    @classmethod
    def _serialize_metrics(
        cls, metrics: dict[str, MetricOrConfig]
    ) -> dict[str, Union[dict, str]]:
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
    def from_json(cls, json_path: Path, **kwargs) -> MetricsHandlerConfig:
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

        return MetricsHandlerConfig(metrics=dict_)

    def get_callable_metrics(self, model: ClinicaDLModel) -> dict[str, Metric]:
        """
        Gets the callable metrics.
        """
        callable_metrics: Dict[str, Metric] = {}

        for name, metric in self.metrics.items():
            if isinstance(metric, LossMetricConfig):
                callable_metrics[name] = metric.get_object(model)
            elif isinstance(metric, MetricConfig):
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
    **metrics : MetricConfig
        Metrics to add to the MetricsHandler. They must be passed as
        :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
    """

    def __init__(
        self,
        **metrics: MetricOrConfig,
    ):
        if not metrics:
            metrics = {}

        self.config = MetricsHandlerConfig(metrics=metrics)
        self._callable_metrics = None
        self._model = None

        self._df = self._init_df()
        self._detailed_df = self._init_detailed_df()

    def init_metrics(self, model: ClinicaDLModel) -> None:
        """
        Instantiates the metrics from their config classes.

        Parameters
        ----------
        model : ClinicaDLModel
            The model that contains the potential losses to compute
            on the validation set.
        """
        self._callable_metrics = self.config.get_callable_metrics(model)
        self._model = model

    @property
    def metrics(self):
        """The metrics currently in the MetricsHandler."""
        return self.config.metrics

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
        **metrics: MetricOrConfig,
    ) -> None:
        """
        Add metrics to the MetricsHandler instance.

        Parameters
        ----------
        **metrics : MetricConfig
            Metrics to add to the MetricsHandler. They must be passed as
            :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
        """
        self.config.add_metrics(metrics)
        if self._callable_metrics is not None:
            self._callable_metrics = self.config.get_callable_metrics(self._model)

        new_columns = self._df.columns.union(self.metrics.keys())
        self._df = self._df.reindex(columns=new_columns, fill_value=pd.NA)

        new_columns = (
            self._detailed_df.columns.drop([PARTICIPANT_ID, SESSION_ID])
            .union(self.metrics.keys())
            .union([PARTICIPANT_ID, SESSION_ID])
        )  # we want participant and session at the end
        self._detailed_df = self._detailed_df.reindex(
            columns=new_columns,
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
        if self._callable_metrics is not None:
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
        if self._callable_metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

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
        if self._callable_metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

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

    def get_metric(self, metric: str, epoch: Optional[int] = None) -> float:
        """
        To get the value of a metric.

        Parameters
        ----------
        metric : str
            The name of the metric.
        epoch : Optional[int], default=None
            The epoch for which the value is wanted. If ``None``, the method will
            return the last computed value.

        Returns
        -------
        float
            The value of the metric.
        """
        if epoch is not None:
            return self.df.set_index(EPOCH).loc[epoch, metric]
        else:
            return self.df.iloc[-1][metric]

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

    def load(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Loads a checkpoint DataFrame saved with :py:meth:`save`.

        Parameters
        ----------
        path : Path
            The path to the DataFrame with the aggregated results.
        details_path: Optional[Path], default=None
            The path to the DataFrame with the detailed results.
            If ``None``, this DataFrame will not be loaded.
        """
        df = pd.read_csv(path, sep=SEP)

        expected_columns = set(self.metrics.keys())
        assert (
            len(expected_columns.difference(df.columns)) == 0
        ), f"Checkpoint in {str(path)} is not a valid metric file, some columns are missing: {expected_columns.difference(df.columns)}"
        self.reset(reset_df=True)
        self._df = df

        if details_path:
            detailed_df = pd.read_csv(details_path, sep=SEP)

            expected_columns = expected_columns.union({PARTICIPANT_ID, SESSION_ID})
            assert (
                len(expected_columns.difference(detailed_df.columns)) == 0
            ), f"Checkpoint in {str(path)} is not a valid metric details file, some columns are missing: {expected_columns.difference(detailed_df.columns)}"
            self._detailed_df = detailed_df

    def write_json(self, json_path: Path) -> None:
        """
        Save the configuration to a JSON file.

        Parameters
        ----------
        json_path : Path
            Destination file path.
        """
        self.config.write_json(json_path)

    @classmethod
    def from_json(
        cls,
        json_path: Path,
        **metrics: MetricConfig,
    ) -> MetricsHandler:
        """
        Creates a MetricsHandler from a JSON file saved with :py:meth:`write_json`.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file.
        **metrics : MetricConfig
            Other metrics to add in the MetricsHandler. It is also a way to pass a custom
            metric that otherwise cannot be read in the JSON file.
        """
        metrics_processor = MetricsHandlerConfig.from_json(json_path, **metrics)
        return cls(**metrics_processor.metrics)
