from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    DATA,
    DATALOADER,
    DATASET,
    TRAIN,
    VALIDATION,
)

from ..base import Directory
from ..utils import SplitsDir


class DataSplitDir(Directory):
    @property
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)


class ValidationDataDir(SplitsDir[DataSplitDir]):
    _dir_type = DataSplitDir

    @property
    def dataset_json(self) -> Path:
        return (self.path / DATASET).with_suffix(JSON)


class TrainDataDir(ValidationDataDir):
    @property
    def dataloader_json(self) -> Path:
        return (self.path / DATALOADER).with_suffix(JSON)


class DataDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)

        self._train = TrainDataDir(path=self.path / TRAIN)
        self._validation = ValidationDataDir(path=self.path / VALIDATION)

    @property
    def train(self) -> TrainDataDir:
        return self._train

    @property
    def validation(self) -> ValidationDataDir:
        return self._validation

    @property
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)
