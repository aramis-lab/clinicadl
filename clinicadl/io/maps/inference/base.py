from __future__ import annotations

from pathlib import Path
from typing import Generic

from clinicadl.utils.dictionary.suffixes import JSON
from clinicadl.utils.dictionary.words import (
    COMPUTATIONAL,
    GROUP,
    MODELS,
    RESULTS,
)

from ..utils import CollectionOfDirs, DataDir, DirType, ModelDir, SplitsDir


class InferenceModelDir(ModelDir):
    @property
    def computational_json(self) -> Path:
        return (self.path / COMPUTATIONAL).with_suffix(JSON)


class InferenceSplitDir(CollectionOfDirs[DirType, str]):
    _item_key = ""
    _separator = ""

    def __init__(self, path: Path):
        super().__init__(path)
        self._models: dict[str, DirType] = {}

    @property
    def models(self) -> dict[str, DirType]:
        return self._models

    @property
    def models_list(self) -> list[str]:
        return self._items_list

    def create_model(
        self, model: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(model, overwrite=overwrite, exist_ok=exist_ok)

    @classmethod
    def _items_dict_private_name(cls) -> str:
        return "_" + MODELS


class InferenceResultsDir(SplitsDir[DirType]):
    pass


class InferenceGroupDir(DataDir, Generic[DirType]):
    _results_dir_type: type[DirType]

    def __init__(self, path: Path):
        super().__init__(path)
        self._results: DirType = self._results_dir_type(path=self.path / RESULTS)

    @property
    def results(self) -> DirType:
        return self._results


class InferenceDir(CollectionOfDirs[DirType, str]):
    _item_key = GROUP

    def __init__(self, path: Path):
        super().__init__(path)
        self._groups: dict[str, DirType] = {}

    @property
    def groups(self) -> dict[str, DirType]:
        return self._groups

    @property
    def groups_list(self) -> list[str]:
        return self._items_list

    def create_group(
        self, group: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(group, overwrite=overwrite, exist_ok=exist_ok)
