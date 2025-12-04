from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.dictionary.words import (
    DATA,
    DATASET,
    GROUP,
    MODELS,
)

from ..utils import CollectionOfDirs, DirType, SplitsDir


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


class InferenceGroupDir(SplitsDir[DirType]):
    @property
    def dataset_json(self) -> Path:
        return (self.path / DATASET).with_suffix(JSON)

    @property
    def data_tsv(self) -> Path:
        return (self.path / DATA).with_suffix(TSV)


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
