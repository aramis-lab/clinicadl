from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from pydantic import Field

from clinicadl.dictionary.utils import SEP
from clinicadl.dictionary.words import (
    EPOCH,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.utils.config import DictOfObjects, KwargsConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.objects import HasConfig

from .base import Metric
from .config import MetricConfig
from .factory import get_metric_from_dict
from .types import MetricOrConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch
    from clinicadl.models import ClinicaDLModel


class MetricsHandlerConfig(KwargsConfig["MetricsHandler"]):
    """
    To check and convert metrics passed by the user.
    """

    metrics: DictOfObjects[Metric, MetricConfig] = Field(
        reader=DictOfObjects.build_reader(get_metric_from_dict)
    )

    def add_metrics(
        self,
        metrics: dict[str, MetricOrConfig],
    ) -> None:
        """
        Adds metrics.
        """
        for name in metrics:
            if name in self.metrics.values:
                raise ValueError(f"A metric named '{name}' already exists!")
        self.metrics = self.metrics.values | metrics

    @classmethod
    def _get_class(cls) -> type[MetricsHandler]:
        """Returns the class associated to this config class."""
        return MetricsHandler


class MetricsHandler(HasConfig[MetricsHandlerConfig]):
    """
    To handle the metrics during a validation phase.

    This object accepts as inputs raw metrics (i.e. objects that inherits from
    :py:class:`clinicadl.metrics.Metric`) or config classes. ``MetricsHandler`` will
    convert config classes to obtain the associated callable.

    ``MetricsHandler`` is itself a callable that works like :py:class:`monai.metricsCumulativeIterationMetric`,
    with :py:meth:`reset` and :py:meth:`aggregate` methods. So, it can be used like a :py:class:`clinicadl.metrics.Metric`
    object.

    The results are stored in DataFrames (:py:attr:`df` and :py:attr:`detailed_df`), that can be saved with
    :py:meth:`save`.

    Parameters
    ----------
    **metrics : MetricConfig
        Metrics to add to the ``MetricsHandler``. They must be passed as
        :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
    """

    _config_type = MetricsHandlerConfig

    def __init__(
        self,
        **metrics: MetricOrConfig,
    ):
        if not metrics:
            metrics = {}

        self.config = MetricsHandlerConfig(metrics=metrics)
        self._metrics = None
        self._model = None

        self._df = self._init_df()
        self._detailed_df = self._init_detailed_df()

    def init_metrics(self, model: Optional[ClinicaDLModel] = None) -> None:
        """
        Instantiates the metrics from their config classes.

        Parameters
        ----------
        model : Optional[ClinicaDLModel], default=None
            The model that contains the potential losses to compute
            on the validation set.
        """
        self._metrics = self.config.metrics.get_object(model=model)
        self._model = model

    @property
    def metrics(self) -> dict[str, Metric]:
        """The metrics currently in the MetricsHandler."""
        return self.config.to_raw_dict()

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
        columns = [PARTICIPANT_ID, SESSION_ID] + list(self.metrics.keys())

        return pd.DataFrame(columns=columns)

    def add_metrics(
        self,
        **metrics: MetricOrConfig,
    ) -> None:
        """
        Add metrics to the MetricsHandler instance.

        .. warning::
            To be sure that all the metrics are computed on the
            same dataset, ``add_metrics`` will reset all the present
            metrics.

        Parameters
        ----------
        **metrics : MetricConfig
            Metrics to add to the MetricsHandler. They must be passed as
            :py:class:`clinicadl.metrics.config.MetricConfig` or :py:class:`clinicadl.metrics.Metric`.
        """
        self.config.add_metrics(metrics)
        self.reset(reset_df=False)
        if self._metrics is not None:
            self._metrics = self.config.metrics.get_object(model=self._model)

        new_columns = self._df.columns.join(self.metrics.keys())
        self._df = self._df.reindex(columns=new_columns, fill_value=pd.NA)

        new_columns = self._detailed_df.columns.join(self.metrics.keys())
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
        if self._metrics is not None:
            for metric in self._metrics.values():
                metric.reset()

        if reset_df:
            self._df = self._init_df()
            self._detailed_df = self._init_detailed_df()

    def aggregate(
        self,
        epoch: Optional[int] = None,
        metrics: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Aggregate and store metric results.

        Parameters
        ----------
        epoch : Optional[int], default=None
            Current epoch. This information will be added in the DataFrame.
        metrics : Optional[Sequence[str]], default=None
            Subset of metrics that must be computed.

        Raises
        ------
        ValueError
            If a metric mentioned in ``metrics`` does not match any metric in the ``MetricsHandler``.

        See Also
        --------
        :py:meth:`monai.metrics.Cumulative.aggregate`
        """
        if self._metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

        to_compute = self._get_metrics_subest(metrics)

        values = {
            name: metric.aggregate()
            for name, metric in self._metrics.items()
            if name in to_compute
        }

        new_df = pd.DataFrame(values, index=[0])

        if epoch is not None:
            new_df.insert(loc=0, column=EPOCH, value=epoch)

        self._df = pd.concat([self._df, new_df], ignore_index=True)

        if epoch is not None:
            self._df.insert(0, EPOCH, self._df.pop(EPOCH))  # ensure epoch first column
            try:
                self._df = self._df.astype(
                    {EPOCH: int}
                )  # type may have been modified by concat
            except pd.errors.IntCastingNaNError:
                pass

    def __call__(
        self,
        batch: Batch,
        epoch: Optional[int] = None,
        metrics: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Updates metrics with a new batch.

        Parameters
        ----------
        batch : Batch
            The batch, with the predictions, and the ground truths if required
            by some metrics.
        epoch : Optional[int], default=None
            Current epoch. This information will be added in the DataFrame.
        metrics : Optional[Sequence[str]], default=None
            Subset of metrics that must be computed.

        Returns
        -------
        pd.DataFrame
            The metrics for all the images in the batch.

        Raises
        ------
        ValueError
            If a metric mentioned in ``metrics`` does not match any metric in the ``MetricsHandler``.
        """
        if self._metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

        to_compute = self._get_metrics_subest(metrics)

        participants = batch.get_field(PARTICIPANT)
        sessions = batch.get_field(SESSION)
        values = {PARTICIPANT_ID: participants, SESSION_ID: sessions}

        values.update(
            {
                name: metric(batch)
                for name, metric in self._metrics.items()
                if name in to_compute
            }
        )

        new_df = pd.DataFrame(values)

        if epoch is not None:
            new_df.insert(loc=0, column=EPOCH, value=epoch)

        self._detailed_df = pd.concat([self._detailed_df, new_df], ignore_index=True)

        if epoch is not None:
            self._detailed_df.insert(
                0, EPOCH, self._detailed_df.pop(EPOCH)
            )  # ensure epoch first column
            try:
                self._detailed_df = self._detailed_df.astype(
                    {EPOCH: int}
                )  # type may have been modified by concat
            except pd.errors.IntCastingNaNError:
                pass

        return new_df

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

    def merge(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Merges the current DataFrame(s) with the one(s) in the file(s) and
        saves the result.

        Parameters
        ----------
        path : Path
            The path for the DataFrame with the aggregated results.
        details_path: Optional[Path], default=None
            The path for the DataFrame with the detailed results.
            If ``None``, this DataFrame will not be saved.
        """
        old_df = pd.read_csv(path, sep=SEP)
        new_df = pd.merge(old_df, self._df, how="outer")
        new_df.to_csv(path, sep=SEP, index=False)

        if details_path:
            old_df = pd.read_csv(details_path, sep=SEP)
            new_df = pd.merge(old_df, self._detailed_df, how="outer")
            new_df.to_csv(details_path, sep=SEP, index=False)

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

    def _get_metrics_subest(self, metrics: Optional[Sequence[str]]) -> Sequence[str]:
        """
        Checks the list of metrics passed.
        """
        if metrics is None:
            return self.metrics
        for metric in metrics:
            if metric not in self.metrics:
                raise ValueError(
                    f"'{metric}' does not match any metrics. Metrics are: {list(self.metrics.keys())}"
                )

        return metrics
