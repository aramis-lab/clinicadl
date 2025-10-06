from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    DATA,
    DATASET,
    GROUP,
    METRICS,
    MODELS,
)

from .base import Directory
from .metrics import MetricsDir
from .utils import CollectionOfDirs


class ModelDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics = MetricsDir(path=self.path / METRICS)

    @property
    def metrics(self) -> MetricsDir:
        return self._metrics


class ModelsDir(CollectionOfDirs[ModelDir, str]):
    _dir_type = ModelDir
    _item_key = ""
    _separator = ""

    def __init__(self, path: Path):
        super().__init__(path)
        self._models: dict[str, ModelDir] = {}

    @property
    def models(self) -> dict[str, ModelDir]:
        return self._models

    @property
    def models_list(self) -> list[int]:
        return self._items_list

    def create_model(
        self, model: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(model, overwrite=overwrite, exist_ok=exist_ok)


class GroupDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._models = ModelsDir(path=self.path / MODELS)

    @property
    def models(self) -> ModelsDir:
        return self._models

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
