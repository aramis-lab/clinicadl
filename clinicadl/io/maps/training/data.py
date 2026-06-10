from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import (
    DATA,
    TRAIN,
    VALIDATION,
)

from ...base import Directory
from ...utils import mandatory
from ..utils import DataDir, SplitsDir


class SplitsDataDir(SplitsDir[DataDir]):
    _dir_type = DataDir


class TrainingDataDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)

        self._train = SplitsDataDir(path=self.path / TRAIN)
        self._validation = SplitsDataDir(path=self.path / VALIDATION)

    @property
    @mandatory
    def train(self) -> SplitsDataDir:
        return self._train

    @property
    @mandatory
    def validation(self) -> SplitsDataDir:
        return self._validation

    @property
    @mandatory
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)
