from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import pandas as pd
from pydantic import PositiveInt, field_validator

from clinicadl.metrics.enum import Optimum
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..base import Callback
from .utils import QuantityMonitoring

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.io.maps.training.splits.models import TrainingModelDir
    from clinicadl.models import Model
    from clinicadl.train import TrainerState


class ModelCheckpointCallbackConfig(ObjectConfig["ModelCheckpointCallback"]):
    """Config class for ``ModelCheckpointCallback``."""

    metric: Optional[str]
    epochs: list[PositiveInt]
    save_last: bool
    mode: Optional[Optimum] = None

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

    save_last : bool, default=True
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
        save_last: bool = True,
    ):
        self.config = self._config_type(
            metric=metric, epochs=epochs, save_last=save_last
        )
        self.metric_monitoring: Optional[QuantityMonitoring] = None
        self._metrics: Optional[MetricsHandler] = None

    def _init_metric_monitoring(self, mode: Optimum) -> None:
        """Initialize metric monitoring with the mode."""
        self.config.mode = mode
        self.metric_monitoring = QuantityMonitoring(
            name=self.config.metric, min_delta=0, mode=mode
        )

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(
        self, *, maps: Maps, state: TrainerState, metrics: MetricsHandler, **kwargs
    ) -> None:
        if self.config.metric:
            metrics.check_metric_name(self.config.metric)
            self._init_metric_monitoring(metrics.metrics[self.config.metric].optimum)
            maps.training.splits[state.split_idx].models.best_models.create_metric(
                metric=self.config.metric, exist_ok=True
            )

    def on_validation_end(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
        **kwargs,
    ) -> None:
        self._metrics = metrics

        if self.config.metric:
            value = metrics.get_metric_value(
                metric=self.config.metric, epoch=state.current_epoch
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
        model_dir: TrainingModelDir,
    ) -> None:
        """
        Saves the model and the validation metrics.
        """
        model_dir.validation_metrics.create(exist_ok=True)
        maps.save_file(model.state_dict(), path=model_dir.model_pt, overwrite=True)
        maps.save_file(
            self._metrics.get_metric_values(epoch=state.current_epoch),
            path=model_dir.validation_metrics.aggregated_tsv,
            overwrite=True,
        )
        maps.save_file(
            self._metrics.get_detailed_metric_values(epoch=state.current_epoch),
            path=model_dir.validation_metrics.details_tsv,
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
