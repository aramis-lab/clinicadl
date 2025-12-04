from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import (
    LOSS,
    TENSORBOARD,
    TRAINING,
)

from ....base import Directory


class LogsDir(Directory):
    @property
    def training_loss(self) -> Path:
        return (self.path / f"{TRAINING}_{LOSS}").with_suffix(TSV)

    @property
    def tensorboard(self) -> Path:
        return self.path / TENSORBOARD
