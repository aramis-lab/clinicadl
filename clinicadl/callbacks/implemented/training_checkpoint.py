from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from pydantic import NonNegativeInt

from clinicadl.train.trainer_state import TrainerState
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import CALLBACKS
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


class TrainingCheckpointCallbackConfig(ObjectConfig["TrainingCheckpointCallback"]):
    """Config class for ``TrainingCheckpointCallback``."""

    every_n_epochs: NonNegativeInt

    @classmethod
    def _get_class(cls):
        return TrainingCheckpointCallback


class TrainingCheckpointCallback(Callback, HasConfig[TrainingCheckpointCallbackConfig]):
    """
    Callback to save model and optimizer checkpoints at specified epochs or intervals.

    This callback copies the current model and optimizer checkpoint files into
    dedicated epoch folders during training, allowing checkpointing at desired points.

    Parameters
    ----------
    patience : int (default=10)
        Interval (in epochs) at which to save checkpoints. For example, if patience=5,
        checkpoints are saved every 5 epochs. The final epoch is always checkpointed.
    epochs : list of int, optional
        Specific epochs at which to save checkpoints regardless of the patience interval.
        If not provided, only the patience interval and the final epoch trigger checkpointing.

    Notes
    -----
    .. note::
        - The final epoch is always saved as a checkpoint.
        - If `patience` is greater than the total number of epochs, it will not save any intermediate checkpoints.
        - If a specific epoch is outside the range of total epochs, it will not raise an error but will not save a checkpoint for that epoch.

    Examples
    --------
    Save checkpoints every 5 epochs:

    .. code-block:: python

        checkpoint = Checkpoint(patience=5)
        checkpoint.on_epoch_end(config=config)


    Save checkpoints at specific epochs 3 and 7, and every 10 epochs:

    .. code-block:: python

        checkpoint = Checkpoint(patience=10, epochs=[3, 7])
        checkpoint.on_epoch_end(config=config)

    """

    _config_type = TrainingCheckpointCallbackConfig

    def __init__(self, every_n_epochs: int):
        self.config = self._config_type(every_n_epochs=every_n_epochs)
        self.last_saved_epoch = 0
        self._metrics = None
        self._callbacks = None

    def on_trainer_init(
        self,
        *,
        metrics: MetricsHandler,
        callbacks: CallbacksHandler,
    ) -> None:
        self._metrics = metrics
        self._callbacks = callbacks

    def on_epoch_end(self, *, model: Model, maps: Maps, state: TrainerState) -> None:
        if state.current_epoch % self.config.every_n_epochs:
            maps.training.splits[state.split_idx].tmp.create_epoch(
                state.current_epoch, overwrite=True
            )
            tmp_dir = maps.training.splits[state.split_idx].tmp.epochs[
                state.current_epoch
            ]

        if (
            config.epoch in self.epochs
            or config.epoch % self.patience == 0
            or config.epoch == config.optim.epochs
        ):
            config.maps.training.splits[config.split.index].checkpoints.create_epoch(
                config.epoch
            )

            epoch_dir = config.maps.training.splits[
                config.split.index
            ].checkpoints.epochs[config.epoch]
            tmp_dir = config.maps.training.splits[config.split.index].tmp.epochs[
                config.epoch
            ]

            shutil.copyfile(tmp_dir.model, epoch_dir.model)

    def _save_callbacks(self, tmp_dir: EpochTmpDir) -> None:
        tmp_dir.callbacks.mkdir()
        names = [
            camel_to_snake(type(callback).__name__)
            for callback in self._callbacks.callbacks
        ]
        for callback in self._callbacks.callbacks:
            path = camel_to_snake(callable)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()
        json_dict.update({"patience": self.patience, "epochs": self.epochs})
        return json_dict
