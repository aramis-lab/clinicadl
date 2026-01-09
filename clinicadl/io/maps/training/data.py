from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.dictionary.words import (
    DATA,
    DATALOADER,
    DATASET,
    TRAIN,
    VALIDATION,
)

from ...base import Directory
from ...utils import mandatory
from ..utils import SplitsDir


class ValidationDataSplitDir(Directory):
    @property
    @mandatory
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)

    @property
    @mandatory
    def dataset_json(self) -> Path:
        return (self.path / DATASET).with_suffix(JSON)


class TrainDataSplitDir(ValidationDataSplitDir):
    @property
    @mandatory
    def dataloader_json(self) -> Path:
        return (self.path / DATALOADER).with_suffix(JSON)


class ValidationDataDir(SplitsDir[ValidationDataSplitDir]):
    _dir_type = ValidationDataSplitDir


class TrainDataDir(SplitsDir[TrainDataSplitDir]):
    _dir_type = TrainDataSplitDir


class DataDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)

        self._train = TrainDataDir(path=self.path / TRAIN)
        self._validation = ValidationDataDir(path=self.path / VALIDATION)

    @property
    @mandatory
    def train(self) -> TrainDataDir:
        return self._train

    @property
    @mandatory
    def validation(self) -> ValidationDataDir:
        return self._validation

    @property
    @mandatory
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)
