from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import (
    LOSS,
    TENSORBOARD,
    TRAINING,
)

from ...base import Directory


class TensorboardDir(Directory):
    pass


class LogsDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._tensorboard = TensorboardDir(path=self.path / TENSORBOARD)

    @property
    def training_loss(self) -> Path:
        return (self.path / f"{TRAINING}_{LOSS}").with_suffix(TSV)

    @property
    def tensorboard(self) -> TensorboardDir:
        return self._tensorboard
