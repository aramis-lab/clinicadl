import shutil
from typing import Optional

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import MODEL, OPTIMIZER
from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class Checkpoint(Callback):
    """Base class for callbacks."""

    def __init__(self, patience: int, epochs: Optional[list[int]] = None):
        self.epochs = epochs if epochs else []
        self.patience = patience

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        if (
            config.epoch in self.epochs
            or config.epoch % self.patience == 0
            or config.epoch == config.optim.epochs
        ):
            config.maps.splits[config.split.index].create_epoch(config.epoch)
            epoch_path = (
                config.maps.splits[config.split.index].epochs[config.epoch].path
            )

            checkpoint_path = config.maps.splits[config.split.index].tmp.checkpoint
            shutil.copyfile(checkpoint_path, epoch_path / (MODEL + PTH + TAR))

            optim_path = config.maps.splits[config.split.index].tmp.optimizer
            shutil.copyfile(optim_path, epoch_path / (OPTIMIZER + PTH + TAR))
