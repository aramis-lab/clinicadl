from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import JSON
from clinicadl.utils.dictionary.words import (
    CALLBACKS,
    DATA,
    OPTIMIZATION,
)

from ..utils import SplitsDir
from .data import DataDir
from .splits import TrainingSplitDir


class TrainingDir(SplitsDir[TrainingSplitDir]):
    _dir_type = TrainingSplitDir

    def __init__(self, path: Path):
        super().__init__(path)
        self._data = DataDir(path=self.path / DATA)

    @property
    def data(self) -> DataDir:
        return self._data

    @property
    def optimization_json(self) -> Path:
        return (self.path / OPTIMIZATION).with_suffix(JSON)

    @property
    def callbacks_json(self) -> Path:
        return (self.path / CALLBACKS).with_suffix(JSON)
