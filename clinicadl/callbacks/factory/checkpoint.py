import shutil
from typing import Any, Optional

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback


class Checkpoint(Callback):
    """
    Callback to save model and optimizer checkpoints at specified epochs or intervals.

    This callback copies the current model and optimizer checkpoint files into
    dedicated epoch folders during training, allowing checkpointing at desired points.

    Parameters
    ----------
    patience : int
        Interval (in epochs) at which to save checkpoints. For example, if patience=5,
        checkpoints are saved every 5 epochs.
    epochs : list of int, optional
        Specific epochs at which to save checkpoints regardless of the patience interval.
        If not provided, only the patience interval and the final epoch trigger checkpointing.

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

    def __init__(self, patience: int, epochs: Optional[list[int]] = None):
        self.epochs = epochs if epochs else []
        self.patience = patience

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        if (
            config.epoch in self.epochs
            or config.epoch % self.patience == 0
            or config.epoch == config.optim.epochs
        ):
            assert config.split is not None
            config.maps.training.splits[config.split.index].checkpoints._create_epoch(
                config.epoch
            )

            epoch_dir = config.maps.training.splits[
                config.split.index
            ].checkpoints.epochs[config.epoch]
            tmp_dir = config.maps.training.splits[config.split.index].tmp

            shutil.copyfile(tmp_dir.model, epoch_dir.model)
            shutil.copyfile(tmp_dir.optimizer, epoch_dir.optimizer)

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
