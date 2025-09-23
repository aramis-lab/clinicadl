from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import PT
from clinicadl.utils.names import camel_to_snake

from .base import Callback

if TYPE_CHECKING:
    from ..handler import _CallbacksHandler


class _CheckpointSaver(Callback):
    """
    Callback that saves the current state of the model and optimizer at the end of each epoch.

    This callback ensures that the training progress is preserved by saving the model weights,
    the optimizer state and current epoch.

    These files are stored in `.pt.tar` format, in the maps, in a temporary directory associated
    with the current training split.

    .. note:
        - This callback is added automatically at the beginning of the training.
        - Used internally for restoring the latest state when training is resumed.

    """

    def __init__(self):
        self._last_saved_epoch = -1

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Check that training and validation DataLoaders are initialized.

        This method ensures that data loading has been configured correctly before training begins.
        """
        if config.split is None:
            raise ValueError(
                "The split has not been initialized. Please run `config.reset(split)`"
            )

        if config.split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if config.split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )

    def on_epoch_end(
        self, config: _TrainingState, callbacks: _CallbacksHandler, **kwargs
    ) -> None:
        """
        Save the current model and optimizer state at the end of each epoch.

        This includes the epoch number and corresponding state dicts for both the
        model and optimizer. These are saved in the `tmp` directory of the current split in the maps.
        """

        if config.epoch == self._last_saved_epoch:
            return
        self._last_saved_epoch = config.epoch

        tmp_dir = config.maps.training.splits[config.split.index].tmp
        tmp_dir._create_epoch(config.epoch)
        epoch_dir = tmp_dir.epochs[config.epoch]

        config.save_checkpoint(epoch_dir)

        for name, callback in callbacks.callbacks.items():
            lowered_name = camel_to_snake(name)
            callback_json = epoch_dir.callbacks / lowered_name
            callback.save_checkpoint(callback_json)

        tmp_dir.clear(except_epoch=config.epoch)

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Remove the temporary storage used for the latest checkpoint after training completes.
        """
        config.maps.training.splits[config.split.index].tmp.remove()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        return self.__dict__
