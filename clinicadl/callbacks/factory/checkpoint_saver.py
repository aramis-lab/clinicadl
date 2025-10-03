from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.utils import SEP
from clinicadl.io import Maps
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.models import ClinicaDLModel
from clinicadl.train.trainer_state import TrainerState
from clinicadl.utils.json import write_json
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
        self_metrics: pd.DataFrame
        self._detailed_metrics: pd.DataFrame

    def on_evaluate_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        metrics: pd.DataFrame,
        detailed_metrics: pd.DataFrame,
    ) -> None:
        self._metrics = metrics
        self._detailed_metrics = detailed_metrics

    def on_epoch_end(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState
    ) -> None:
        """
        Save the current model and optimizer state at the end of each epoch.

        This includes the epoch number and corresponding state dicts for both the
        model and optimizer. These are saved in the `tmp` directory of the current split in the maps.
        """

        if state.current_epoch == self._last_saved_epoch:
            return
        self._last_saved_epoch = state.current_epoch

        tmp_dir = maps.training.splits[state.split_idx].tmp
        tmp_dir.read()
        tmp_dir.create_epoch(state.current_epoch)
        epoch_dir = tmp_dir.epochs[state.current_epoch]

        # model
        model.save_checkpoint(epoch_dir.model)

        # metrics
        self._metrics.to_csv(
            epoch_dir.validation_metrics.aggregated, sep=SEP, index=False
        )
        self._detailed_metrics.to_csv(
            epoch_dir.validation_metrics.details, sep=SEP, index=False
        )

        # trainer state
        write_json(epoch_dir, state.state_dict())

        # callbacks
        for name, callback in callbacks.callbacks.items():
            lowered_name = camel_to_snake(name)
            callback_json = epoch_dir.callbacks / lowered_name
            callback.save_checkpoint(callback_json)

        # delete old epochs
        for epoch in tmp_dir.epochs_list:
            if epoch != state.current_epoch:
                tmp_dir.epochs[epoch].remove()

    def on_train_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """
        Remove the temporary storage used for the latest checkpoint after training completes.
        """
        tmp_dir = maps.training.splits[state.split_idx].tmp
        for epoch in tmp_dir.epochs_list:
            tmp_dir.epochs[epoch].remove()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        return self.__dict__
