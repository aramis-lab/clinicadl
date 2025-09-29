from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import (
    LOGS,
    TENSORBOARD,
    TRAINING,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


class LogsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / LOGS)

    @property
    def training_tsv(self) -> Path:
        return self.path / (TRAINING + TSV)

    @property
    def tensorboard(self) -> Path:
        return self.path / TENSORBOARD
