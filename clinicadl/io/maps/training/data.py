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


class DataSplitDir(Directory):
    @property
    @mandatory
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)

    @property
    @mandatory
    def dataset_json(self) -> Path:
        return (self.path / DATASET).with_suffix(JSON)

    @property
    @mandatory
    def dataloader_json(self) -> Path:
        return (self.path / DATALOADER).with_suffix(JSON)


class DataDir(SplitsDir[DataSplitDir]):
    _dir_type = DataSplitDir


class TrainingDataDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)

        self._train = DataDir(path=self.path / TRAIN)
        self._validation = DataDir(path=self.path / VALIDATION)

    @property
    @mandatory
    def train(self) -> DataDir:
        return self._train

    @property
    @mandatory
    def validation(self) -> DataDir:
        return self._validation

    @property
    @mandatory
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)
