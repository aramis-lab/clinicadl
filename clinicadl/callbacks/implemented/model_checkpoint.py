from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import pandas as pd
from pydantic import PositiveInt, field_validator, model_validator
from typing_extensions import Self

from clinicadl.metrics.enum import Optimum
from clinicadl.train.trainer_state import TrainerCall
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..base import Callback
from .utils import (
    QuantityMonitoring,
    build_metric_key_error,
    get_metric_value,
    get_metric_values,
)

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.io.maps.training.splits.models import ModelDir
    from clinicadl.metrics import Metric
    from clinicadl.models import Model
    from clinicadl.train import TrainerState


class ModelCheckpointCallbackConfig(ObjectConfig["ModelCheckpointCallback"]):
    """Config class for ``ModelCheckpointCallback``."""

    metric: Optional[str]
    epochs: list[PositiveInt]
    save_last: Optional[bool]
    mode: Optional[Optimum] = None

    @model_validator(mode="after")
    def _validate_save_last(self) -> Self:
        """If no monitored quantity and save_last is None, set save_last to True."""
        if not self.metric and self.save_last is None:
            self.__dict__["save_last"] = True

        return Self

    @field_validator("epochs", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        """Converts None to an empty list for 'epochs'."""
        if value is None:
            return []
        return value

    @classmethod
    def _get_class(cls):
        return ModelCheckpointCallback


class ModelCheckpointCallback(Callback, HasConfig[ModelCheckpointCallbackConfig]):
    """
    To save checkpoints of the neural network weights at different point of the training.

    Checkpoints can be saved after specified epochs and/or according to a monitored
    metric. In the latter case, only the best model according to this metric will be saved.
    The neural network weights after the last epoch can also be saved.

    Parameters
    ----------
    metric : Optional[str], default=None
        The metric to monitor.
    epochs : Optional[Sequence[int]], default=None
        The list of epochs after which the neural network weights should be saved.

        .. important::
            Epochs are indexed from **1**.

    save_last : bool
        Whether to save the neural network weights after the last epoch.

    Examples
    --------
    .. code-block::

        from clinicadl.callbacks import ModelCheckpointCallback
        from clinicadl.train import Trainer
        from clinicadl.metrics.config import MSEMetricConfig, LossMetricConfig
        ...

        trainer = Trainer(
            metrics={"loss": LossMetricConfig(), "mse": MSEMetricConfig()},
            callbacks=[ModelCheckpointCallback(metric="mse", epochs=range(1, 100, step=10), save_last=True)],
            ...
        )
    """

    _config_type = ModelCheckpointCallbackConfig

    def __init__(
        self,
        metric: Optional[str] = None,
        epochs: Optional[Sequence[int]] = None,
        save_last: Optional[bool] = None,
    ):
        self.config = self._config_type(
            metric=metric, epochs=epochs, save_last=save_last
        )
        self.metric_monitoring: Optional[QuantityMonitoring] = None
        self._metrics_df = pd.DataFrame()
        self._detailed_metrics_df = pd.DataFrame()

    def _init_metric_monitoring(self, mode: Optimum) -> None:
        """Initialize metric monitoring with the mode."""
        self.config.mode = mode
        self.metric_monitoring = QuantityMonitoring(
            name=self.config.metric, min_delta=0, mode=mode
        )

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(self, *, maps: Maps, state: TrainerState, **kwargs) -> None:
        if self.config.metric:
            maps.training.splits[state.split_idx].models.best_models.create_metric(
                metric=self.config.metric, exist_ok=True
            )

        if self.metric_monitoring:
            self.metric_monitoring.reset()

    def on_validation_start(
        self, *, state: TrainerState, metrics: dict[str, Metric], **kwargs
    ) -> None:
        if (
            self.config.metric
            and self.metric_monitoring is None
            and state.called == TrainerCall.TRAIN
        ):
            try:
                mode = metrics[self.config.metric].optimum
            except KeyError as exc:
                raise build_metric_key_error(self.config.metric) from exc

            self._init_metric_monitoring(mode)

    def on_validation_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics_df: pd.DataFrame,
        detailed_metrics_df: pd.DataFrame,
        **kwargs,
    ) -> None:
        if state.called != TrainerCall.TRAIN:
            return

        self._metrics_df = metrics_df
        self._detailed_metrics_df = detailed_metrics_df

        if self.config.metric:
            value = get_metric_value(
                metrics_df, metric_name=self.config.metric, epoch=state.current_epoch
            )

            if self.metric_monitoring.step(value, log=False):
                model_dir = maps.training.splits[
                    state.split_idx
                ].models.best_models.metrics[self.config.metric]

                self._save_files(model, maps, state, model_dir)

    def on_epoch_end(
        self, *, model: Model, maps: Maps, state: TrainerState, **kwargs
    ) -> None:
        if state.current_epoch in self.config.epochs:
            maps.training.splits[state.split_idx].models.checkpoints.create_epoch(
                state.current_epoch
            )
            model_dir = maps.training.splits[state.split_idx].models.checkpoints.epochs[
                state.current_epoch
            ]

            self._save_files(model, maps, state, model_dir)

    def on_train_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        **kwargs,
    ) -> None:
        if self.config.save_last:
            model_dir = maps.training.splits[state.split_idx].models.final
            self._save_files(model, maps, state, model_dir)

    def state_dict(self) -> Mapping[str, Any]:
        return self.metric_monitoring.state_dict() if self.metric_monitoring else {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict and self.metric_monitoring:
            self.metric_monitoring.load_state_dict(state_dict)

    def _save_files(
        self,
        model: Model,
        maps: Maps,
        state: TrainerState,
        model_dir: ModelDir,
    ) -> None:
        """
        Saves the model and the validation metrics.
        """
        model_dir.validation_metrics.create(exist_ok=True)
        maps.save_file(model.state_dict(), path=model_dir.model, overwrite=True)
        maps.save_file(
            get_metric_values(self._metrics_df, epoch=state.current_epoch),
            path=model_dir.validation_metrics.aggregated,
            overwrite=True,
        )
        maps.save_file(
            get_metric_values(self._detailed_metrics_df, epoch=state.current_epoch),
            path=model_dir.validation_metrics.details,
            overwrite=True,
        )

    @classmethod
    def _from_config(cls, config):
        args = config.to_raw_dict()
        mode = args.pop("mode")

        early_stopper = cls(**args)

        if mode:
            early_stopper._init_metric_monitoring(mode)

        return early_stopper
