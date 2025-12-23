import shutil
from typing import Any, Optional

from clinicadl.train.trainer_state import TrainerState

from ..base import Callback

"""
.. note::

    Behavior regarding interaction with ``ModelSelection``:

    - If neither ``EarlyStopping`` nor ``ModelSelection`` are used:
        the final model and the best-loss model are saved, but no early stopping is applied.
    - If ``EarlyStopping`` is used without ``ModelSelection``:
        training stops when all monitored metrics stop improving.
        For each metric, a ``ModelSelection`` object is automatically created.
    - If ``ModelSelection`` is used without ``EarlyStopping``:
        best models are saved based on monitored metrics, but training completes all epochs.
    - If both are used:
        ``EarlyStopping`` metrics are automatically tracked by ``ModelSelection``,
        ensuring best-performing models are saved.
"""


class Checkpoint(Callback):
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

    def __init__(self, patience: int = 10, epochs: Optional[list[int]] = None):
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")

        if not isinstance(epochs, list) and isinstance(epochs, int):
            epochs = [epochs]

        self.epochs = epochs if epochs else []
        self.patience = patience

    def on_epoch_end(self, config: TrainerState, **kwargs) -> None:
        """
        Save the current model and optimizer state at the end of the require epochs.
        It needs to be called after the _CheckpointSaver callback so that the tmp files
        exist and are up to date.
        """
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
