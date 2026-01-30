from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import NonNegativeInt

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.suffixes import PT
from clinicadl.utils.names import camel_to_snake
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.io.maps.training.splits.tmp import EpochTmpDir
    from clinicadl.metrics import MetricsHandler
    from clinicadl.models import Model
    from clinicadl.train import TrainerState

    from ..handler import CallbacksHandler

logger = logging.getLogger("clinicadl.callbacks.TrainingCheckpointCallback")


class TrainingCheckpointCallbackConfig(ObjectConfig["TrainingCheckpointCallback"]):
    """Config class for ``TrainingCheckpointCallback``."""

    every_n_epochs: NonNegativeInt
    enabled: bool

    @classmethod
    def _get_class(cls):
        return TrainingCheckpointCallback


class TrainingCheckpointCallback(Callback, HasConfig[TrainingCheckpointCallbackConfig]):
    """
    To save checkpoints during a training phase.

    The user can then resume a training from the last saved checkpoint when calling
    :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.

    The checkpoints will be **deleted when the training is completed**. To save permanently
    checkpoints of your neural network, use instead :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`.

    Parameters
    ----------
    every_n_epochs : int, default=10
        Interval (in epochs) for saving checkpoints.
    enabled : bool, default=True
        Whether to activate checkpointing.
    """

    _config_type = TrainingCheckpointCallbackConfig

    def __init__(self, every_n_epochs: int = 10, enabled: bool = True):
        self.config = self._config_type(every_n_epochs=every_n_epochs, enabled=enabled)
        self._metrics = None
        self._callbacks = None
        self._optimizers = None
        self._scaler = None
        self._metrics = None

    def on_trainer_init(
        self,
        *,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
        **kwargs,
    ) -> None:
        self._metrics = metrics
        self._callbacks = callbacks

    def on_exception(self, *, maps: Maps, state: TrainerState, **kwargs) -> None:
        if self.config.enabled:
            last_saved_epoch = self._get_last_saved_epoch(maps, state.split_idx)
            logger.error("Last checkpoint at the end of epoch %d", last_saved_epoch)

    def on_resume(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
        **kwargs,
    ) -> None:
        tmp_dir = maps.training.splits[state.split_idx].tmp
        last_saved_epoch = self._get_last_saved_epoch(maps, split_idx=state.split_idx)
        logger.info("Loading checkpoints from epoch %d", last_saved_epoch)
        chkpt_dir = tmp_dir.epochs[last_saved_epoch]

        state.load_state_dict(maps.open_file(chkpt_dir.state_json))
        model.load_state_dict(maps.open_file(chkpt_dir.model_pt))
        opt_state_dicts = maps.open_file(chkpt_dir.optimizer_pt)
        for opt_name, opt in optimizers.items():
            opt.load_state_dict(opt_state_dicts[opt_name])
        grad_scaler.load_state_dict(maps.open_file(chkpt_dir.scaler_pt))
        metrics.load(
            chkpt_dir.validation_metrics.aggregated_tsv,
            details_path=chkpt_dir.validation_metrics.details_tsv,
        )
        self._load_callbacks(callbacks, chkpt_dir=chkpt_dir, maps=maps)

    def on_optimization_step_end(
        self,
        *,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
        **kwargs,
    ) -> None:
        self._optimizers = optimizers
        self._scaler = grad_scaler

    def on_epoch_end(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        if (
            not self.config.enabled
            or not state.current_epoch % self.config.every_n_epochs == 0
        ):
            return

        maps.training.splits[state.split_idx].tmp.create_epoch(
            state.current_epoch, overwrite=True
        )
        chkpt_dir = maps.training.splits[state.split_idx].tmp.epochs[
            state.current_epoch
        ]

        state.to_json(chkpt_dir.state_json)
        maps.save_file(model.state_dict(), chkpt_dir.model_pt)
        maps.save_file(
            {opt_name: opt.state_dict() for opt_name, opt in self._optimizers.items()},
            chkpt_dir.optimizer_pt,
        )
        maps.save_file(self._scaler.state_dict(), chkpt_dir.scaler_pt)
        self._metrics.save(
            chkpt_dir.validation_metrics.aggregated_tsv,
            details_path=chkpt_dir.validation_metrics.details_tsv,
        )
        self._save_callbacks(chkpt_dir, maps=maps)

        maps.training.splits[state.split_idx].tmp.clear(
            except_epoch=state.current_epoch
        )
        logger.debug("Training checkpoint saved after epoch %d", state.current_epoch)

    def on_train_end(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        **kwargs,
    ) -> None:
        if maps.training.splits[state.split_idx].tmp.path.exists():
            maps.training.splits[state.split_idx].tmp.clear()

    def _save_callbacks(self, chkpt_dir: EpochTmpDir, maps: Maps) -> None:
        """
        Saves the callback checkpoints.
        """
        for callback, file_name in zip(
            self._callbacks.callbacks,
            self._get_callback_file_names(self._callbacks.callbacks),
        ):
            maps.save_file(callback.state_dict(), chkpt_dir.callbacks / file_name)

    @classmethod
    def _load_callbacks(
        cls, callbacks: CallbacksHandler, chkpt_dir: EpochTmpDir, maps: Maps
    ) -> None:
        """
        Loads the callback checkpoints.
        """
        for callback, file_name in zip(
            callbacks.callbacks, cls._get_callback_file_names(callbacks.callbacks)
        ):
            callback.load_state_dict(maps.open_file(chkpt_dir.callbacks / file_name))

    @staticmethod
    def _get_callback_file_names(callbacks: list[Callback]) -> list[Path]:
        """
        Gives a snake case file name for each callback.
        """
        names = [camel_to_snake(type(callback).__name__) for callback in callbacks]
        cnt = {name: 0 for name in set(names)}
        for i, name in enumerate(names):
            names[i] += f"_{cnt[name]}" if cnt[name] else ""
            cnt[name] += 1

        return map(lambda x: Path(x).with_suffix(PT), names)

    @staticmethod
    def _get_last_saved_epoch(maps: Maps, split_idx: int) -> int:
        """
        Gets the last epoch saved in the checkpoint directory.
        """
        tmp_dir = maps.training.splits[split_idx].tmp
        tmp_dir.read()
        try:
            return int(sorted(tmp_dir.epochs_list)[-1])
        except IndexError as e:
            raise FileNotFoundError(
                f"No training checkpoint found in {str(tmp_dir.path)}"
            ) from e
