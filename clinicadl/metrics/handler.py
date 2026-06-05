from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import pandas as pd
from pydantic import Field, ValidationError, ValidationInfo, field_validator
from typing_extensions import Self

from clinicadl.utils.config import DictOfObjects, ObjectConfig
from clinicadl.utils.dictionary.utils import TSV_SEP
from clinicadl.utils.dictionary.words import (
    CPU,
    EPOCH,
    METRICS,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.utils.exceptions import CannotReadFieldError, ClinicaDLConfigurationError
from clinicadl.utils.objects import HasConfig

from .base import Metric
from .config import MetricConfig
from .factory import get_metric_from_dict
from .types import MetricOrConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch
    from clinicadl.models import Model


class MetricsHandlerConfig(ObjectConfig["MetricsHandler"]):
    """
    To check and convert metrics passed by the user.
    """

    metrics: DictOfObjects[Metric, MetricConfig] = Field(
        reader=DictOfObjects.build_reader(get_metric_from_dict)
    )
    metrics_on_cpu: bool

    @field_validator("metrics", mode="before")
    @classmethod
    def _handle_dict(cls, v: Any, info: ValidationInfo) -> DictOfObjects:
        return DictOfObjects.from_dict(v, field_name=info.field_name)

    @classmethod
    def from_dict(cls, dict_: dict[str, Any], **kwargs) -> Self:
        dict_ = cls._check_dict(dict_)

        dict_[METRICS].update(
            {arg: value for arg, value in kwargs.items() if arg != "metrics_on_cpu"}
        )

        if cpu := kwargs.get("metrics_on_cpu", None):
            dict_["metrics_on_cpu"] = cpu

        try:
            return super().from_dict(dict_)
        except CannotReadFieldError as e:
            if wrong_metrics := cls._read_field_reading_error(e):
                raise CannotReadFieldError(
                    field_names=wrong_metrics,
                    object_name=cls._get_name(),
                    error=e.error,
                ) from e
            raise

    @property
    def metric_names(self) -> list[str]:
        """
        The names of the metrics.
        """
        return list(self.metrics.values.keys())

    def add_metrics(
        self,
        metrics: dict[str, MetricOrConfig],
    ) -> None:
        """
        Adds metrics.
        """
        for name in metrics:
            if name in self.metric_names:
                raise ValueError(f"A metric named '{name}' already exists!")
        self.metrics = self.metrics.values | metrics

    def remove_metrics(
        self,
        metrics: Sequence[str],
    ) -> None:
        """
        Removes metrics.
        """
        for metric in metrics:
            self.metrics.values.pop(metric)

    @staticmethod
    def _read_field_reading_error(error: CannotReadFieldError) -> Optional[list[str]]:
        """
        To get the potential metrics that are causing troubles.
        """
        if isinstance(error.error, CannotReadFieldError):
            return error.error.field_names
        elif isinstance(error.error, ValidationError):
            return sorted(list(set(e["loc"][2] for e in error.error.errors())))

    @classmethod
    def _get_class(cls) -> type[MetricsHandler]:
        """Returns the class associated to this config class."""
        return MetricsHandler


class MetricsHandler(HasConfig[MetricsHandlerConfig]):
    """
    To handle all the :py:class:`~clinicadl.metrics.Metric` computed during an evaluation phase.

    ``MetricsHandler`` is itself a callable that works like :py:class:`monai.metrics.CumulativeIterationMetric`,
    with :py:meth:`reset` and :py:meth:`aggregate` methods. So, it can be used like a raw :py:class:`~clinicadl.metrics.Metric`
    object.

    The results are stored in DataFrames (:py:attr:`df` and :py:attr:`detailed_df`), that can be saved with
    :py:meth:`save`.

    Parameters
    ----------
    metrics_on_cpu : boo, True
        Whether to necessarily apply metrics computation on CPU. If ``False``, postprocessing will
        be applied on the device where are currently the data
    **metrics : MetricOrConfig
        Metrics to add to the ``MetricsHandler``. They must be passed as :py:class:`~clinicadl.metrics.Metric`
        or :py:mod:`configuration classes <clinicadl.metrics.config>`.

    Examples
    --------

    .. code-block::

        from clinicadl.metrics import MetricsHandler, config
        from clinicadl.data.structures.examples import Colin27DataPoint
        from clinicadl.data.dataloader import Batch
        import torch

        metrics = MetricsHandler(
            mse=config.MSEMetricConfig(label_key="ground_truth"),
            mae=config.MAEMetricConfig(label_key="ground_truth"),
        )
        metrics.init_metrics()

        torch.manual_seed(0)
        batch = Batch(
            [
                Colin27DataPoint(output=torch.randn(1, 10), ground_truth=torch.randn(1, 10)),
                Colin27DataPoint(output=torch.randn(1, 10), ground_truth=torch.randn(1, 10)),
            ]
        )

    .. code-block::

        >>> metrics(batch)
           epoch participant_id session_id       mse       mae
        0      1        sub-000   ses-M000  1.155020  0.871509
        1      1        sub-001   ses-M000  1.540214  0.835756
        >>> metrics.detailed_df
           epoch participant_id session_id       mse       mae
        0      1        sub-000   ses-M000  1.155020  0.871509
        1      1        sub-001   ses-M000  1.540214  0.835756
        >>> metrics.aggregate(epoch=1)
        >>> metrics.df
           epoch       mse       mae
        0      1  1.347617  0.853633

    """

    _config_type = MetricsHandlerConfig

    def __init__(
        self,
        metrics_on_cpu: bool = True,
        **metrics: MetricOrConfig,
    ):
        if not metrics:
            metrics = {}

        self.config = MetricsHandlerConfig(
            metrics_on_cpu=metrics_on_cpu, metrics=metrics
        )
        self._metrics = None
        self._model = None

        self._df = self._init_df()
        self._detailed_df = self._init_detailed_df()

    def init_metrics(self, model: Optional[Model] = None) -> None:
        """
        Instantiates the metrics from their configuration classes.

        Parameters
        ----------
        model : Optional[Model], default=None
            The model that contains the potential losses to compute
            on the validation set (defined in :py:meth:`Model.get_loss_functions <clinicadl.models.Model.get_loss_functions>`).
        """
        self._metrics = self.config.metrics.get_object(model=model)
        self._model = model

    @property
    def metrics(self) -> Optional[dict[str, Metric]]:
        """
        The metrics currently in the ``MetricsHandler``.
        If ``None``, it means that :py:meth:`init_metrics` must be called.
        """
        return self._metrics

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
        return pd.DataFrame(columns=self.config.metric_names)

    def _init_detailed_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with a column for each metric,
        as well as columns "participant_id" and "session_id".
        """
        columns = [PARTICIPANT_ID, SESSION_ID] + self.config.metric_names

        return pd.DataFrame(columns=columns)

    def add_metrics(
        self,
        **metrics: MetricOrConfig,
    ) -> None:
        """
        Adds metrics to the ``MetricsHandler`` instance.

        .. warning::
            To be sure that all the metrics are computed on the
            same dataset, ``add_metrics`` will reset all the present
            metrics.

        Parameters
        ----------
        **metrics : MetricConfig
            Metrics to add to the ``MetricsHandler``. They must be passed as :py:class:`~clinicadl.metrics.Metric`
            or :py:mod:`configuration classes <clinicadl.metrics.config>`.
        """
        self.config.add_metrics(metrics)
        self.reset(reset_df=False)
        if self.metrics is not None:
            self._metrics = self.config.metrics.get_object(model=self._model)

        new_columns = self._df.columns.join(self.config.metric_names)
        self._df = self._df.reindex(columns=new_columns, fill_value=pd.NA)

        new_columns = self._detailed_df.columns.join(self.config.metric_names)
        self._detailed_df = self._detailed_df.reindex(
            columns=new_columns,
            fill_value=pd.NA,
        )

    def remove_metrics(
        self,
        metrics: Union[str, Sequence[str]],
    ) -> None:
        """
        Removes metrics from the ``MetricsHandler`` instance.

        Parameters
        ----------
        metrics : Union[str, Sequence[str]]
            The name of the metric(s) to remove.
        """
        if isinstance(metrics, str):
            metrics = [metrics]

        self.config.remove_metrics(metrics)
        if self.metrics is not None:
            for metric in metrics:
                self._metrics.pop(metric)

        self._df.drop(columns=metrics, inplace=True)
        self._detailed_df.drop(columns=metrics, inplace=True)

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
        if self.metrics is not None:
            for metric in self.metrics.values():
                metric.reset()

        if reset_df:
            self._df = self._init_df()
            self._detailed_df = self._init_detailed_df()

    def aggregate(
        self,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Aggregates and stores metric results.

        Parameters
        ----------
        epoch : Optional[int], default=None
            Current epoch. This information will be added in the DataFrame.

        See Also
        --------
        :py:meth:`monai.metrics.Cumulative.aggregate`
        """
        if self.metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

        values = {name: metric.aggregate() for name, metric in self.metrics.items()}

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

        Returns
        -------
        pd.DataFrame
            The metrics for all the images in the batch.
        """
        if self.metrics is None:
            raise ClinicaDLConfigurationError(
                "First, call 'init_metrics' to instantiate the metrics."
            )

        if self.config.metrics_on_cpu:
            batch.to(device=CPU)

        participants = batch.get_field(PARTICIPANT)
        sessions = batch.get_field(SESSION)
        values = {PARTICIPANT_ID: participants, SESSION_ID: sessions}

        values.update({name: metric(batch) for name, metric in self.metrics.items()})

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

    def get_metric_value(self, metric: str, epoch: Optional[int] = None) -> float:
        """
        To get the (aggregated) value of a metric.

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
        self.check_metric_name(metric)

        if epoch is not None:
            value = self.df.set_index(EPOCH).loc[epoch, metric]
        else:
            value = self.df.iloc[-1][metric]

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"""Value for metric '{metric}' {f"at epoch {epoch} " if epoch is not None else ""}is not numeric."""
            ) from exc

    def get_metric_values(self, epoch: Optional[int] = None) -> pd.DataFrame:
        """
        To get the (aggregated) values of all the computed metrics.

        Parameters
        ----------
        epoch : int
            The epoch for which the values are wanted. If ``None``, the method will
            return the last computed values.

        Returns
        -------
        pd.DataFrame
            A :py:class:`pandas.DataFrame` containing the metric values.
        """
        if epoch is not None:
            return self.df[self.df[EPOCH] == epoch]
        return self.df.iloc[-1:]

    def get_detailed_metric_values(self, epoch: int) -> pd.DataFrame:
        """
        To get the detailed values of a all the computed metrics
        for a specific epoch.

        Parameters
        ----------
        epoch : int
            The epoch for which the values are wanted.

        Returns
        -------
        pd.DataFrame
            A :py:class:`pandas.DataFrame` containing the metric values.
        """
        return self.detailed_df[self.detailed_df[EPOCH] == epoch]

    def check_metric_name(self, metric: str) -> None:
        """
        Checks if a metric is in the computed metrics.

        Parameters
        ----------
        metric : str
            The name of the metric to check.
        """
        if metric not in self.config.metric_names:
            raise KeyError(
                f"'{metric}' not found in the computed metrics! Metrics are: {self.config.metric_names}"
            )

    def save(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Saves the DataFrames containing the results.

        Parameters
        ----------
        path : Path
            The path where to save :py:attr:`df`.
        details_path: Optional[Path], default=None
            The path where to save :py:attr:`detailed_df`.
            If ``None``, this DataFrame will not be saved.
        """
        self._df.to_csv(path, sep=TSV_SEP, index=False)
        if details_path:
            self._detailed_df.to_csv(details_path, sep=TSV_SEP, index=False)

    def merge(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Merges the current DataFrame(s) with the one(s) in the file(s) and
        saves the result.

        Parameters
        ----------
        path : Path
            The path to the DataFrame to merge with :py:attr:`df`.
        details_path: Optional[Path], default=None
            The path to the DataFrame to merge with :py:attr:`detailed_df`.
        """
        old_df = pd.read_csv(path, sep=TSV_SEP)
        try:
            new_df = pd.merge(old_df, self._df, how="outer")
        except pd.errors.MergeError:
            new_df = pd.concat([old_df, self._df], axis=1)
        new_df.to_csv(path, sep=TSV_SEP, index=False)

        if details_path:
            old_df = pd.read_csv(details_path, sep=TSV_SEP)
            new_df = pd.merge(old_df, self._detailed_df, how="outer")
            new_df.to_csv(details_path, sep=TSV_SEP, index=False)

    def load(self, path: Path, details_path: Optional[Path] = None) -> None:
        """
        Loads checkpoint DataFrames saved with :py:meth:`save`.

        Parameters
        ----------
        path : Path
            The path to the DataFrame with the aggregated results.
        details_path: Optional[Path], default=None
            The path to the DataFrame with the detailed results.
            If ``None``, this DataFrame will not be loaded.
        """
        df = pd.read_csv(path, sep=TSV_SEP)

        expected_columns = set(self.config.metric_names)
        assert (
            len(expected_columns.difference(df.columns)) == 0
        ), f"Checkpoint in {str(path)} is not a valid metric file, some columns are missing: {expected_columns.difference(df.columns)}"
        self.reset(reset_df=True)
        self._df = df

        if details_path:
            detailed_df = pd.read_csv(details_path, sep=TSV_SEP)

            expected_columns = expected_columns.union({PARTICIPANT_ID, SESSION_ID})
            assert (
                len(expected_columns.difference(detailed_df.columns)) == 0
            ), f"Checkpoint in {str(path)} is not a valid metric details file, some columns are missing: {expected_columns.difference(detailed_df.columns)}"
            self._detailed_df = detailed_df

    def subset(self, metrics: Sequence[str]) -> MetricsHandler:
        """
        Creates a new ``MetricsHandler`` containing only a subset of
        the current metrics.

        Parameters
        ----------
        metrics : Sequence[str]
            The names of the metrics to keep in the new instance.

        Returns
        -------
        MetricsHandler
            The new instance.
        """
        for metric in metrics:
            self.check_metric_name(metric)

        subset = {
            name: metric
            for name, metric in self.config.to_raw_dict()[METRICS].items()
            if name in metrics
        }

        new_metrics = MetricsHandler(
            **deepcopy(subset), metrics_on_cpu=self.config.metrics_on_cpu
        )
        if self.metrics:
            new_metrics.init_metrics(model=self._model)

        return new_metrics

    @classmethod
    def _from_config(cls: type[Self], config: MetricsHandlerConfig) -> Self:
        """To create the object from the associated config."""
        args = config.to_raw_dict()
        return cls(metrics_on_cpu=args["metrics_on_cpu"], **args[METRICS])
