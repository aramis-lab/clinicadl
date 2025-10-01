from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    DATA,
    DATASET,
    GROUP,
    METRICS,
)

from .base import Directory
from .metrics import MetricsDir
from .utils import BestModelsDir, CollectionOfDirs, SplitsDir


class PredictionsMetrics(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics = MetricsDir(path=self.path / METRICS)

    @property
    def metrics(self) -> MetricsDir:
        return self._metrics


class PredictionsSplitDir(BestModelsDir[PredictionsMetrics]):
    _dir_type = PredictionsMetrics


class GroupDir(SplitsDir[PredictionsSplitDir]):
    _dir_type = PredictionsSplitDir

    @property
    def dataset_json(self) -> Path:
        return (self.path / DATASET).with_suffix(JSON)

    @property
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)


class PredictionsDir(CollectionOfDirs[GroupDir, str]):
    _dir_type = GroupDir
    _item_key = GROUP

    def __init__(self, path: Path):
        super().__init__(path)
        self._groups: dict[str, GroupDir] = {}

    @property
    def groups(self) -> dict[str, GroupDir]:
        return self._groups

    @property
    def groups_list(self) -> list[int]:
        return self._items_list

    def create_group(
        self, group: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(group, overwrite=overwrite, exist_ok=exist_ok)
