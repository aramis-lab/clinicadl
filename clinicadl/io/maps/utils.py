from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, Generic, TypeVar

from clinicadl.dictionary.words import BEST, EPOCH, METRICS, SPLIT

from ..base import Directory

DirType = TypeVar("DirType", bound=Directory)
ItemType = TypeVar("ItemType", int, str)


class CollectionOfDirs(Generic[DirType, ItemType], Directory):
    _dir_type: type[DirType]
    _item_key: str
    _item_mapping: Callable[[str], ItemType] = staticmethod(lambda x: x)
    _separator = "-"

    @property
    def _items_list(self) -> list[ItemType]:
        dict_: dict = getattr(self, self._items_dict_private_name())
        return sorted(list(dict_.keys()))

    def read(self):
        self._find_item_dirs()
        super().read()

    def _find_item_dirs(self) -> None:
        if self._separator:
            items = [
                x.name.split(self._separator)[-1]
                for x in self.path.iterdir()
                if x.name.startswith(self._item_key)
            ]
        else:
            items = [
                x.name for x in self.path.iterdir() if x.name.startswith(self._item_key)
            ]
        sub_dirs = {
            self._item_mapping(item): self._dir_type(self._item_path(item))
            for item in items
        }
        setattr(
            self,
            self._items_dict_private_name(),
            sub_dirs,
        )

    def _item_path(self, item: str) -> Path:
        return self.path / f"{self._item_key}{self._separator}{item}"

    def _create_item(
        self, item: ItemType, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        dir_: DirType = self._dir_type(self._item_path(str(item)))
        dir_.create(overwrite=overwrite, exist_ok=exist_ok)
        dict_ = getattr(self, self._items_dict_private_name())
        dict_[item] = dir_

    @classmethod
    def _items_dict_private_name(cls) -> str:
        return "_" + cls._item_key + "s"

    def iterdir(self) -> Generator[Directory, None, None]:
        """
        To iterate over the Directories of the collections.

        Returns
        -------
        Generator[Directory, None, None]
        """
        for item in self._items_list:
            yield getattr(self, self._items_dict_private_name())[item]


class SplitsDir(CollectionOfDirs[DirType, int]):
    _item_key = SPLIT
    _item_mapping = staticmethod(lambda x: int(x))

    def __init__(self, path: Path):
        super().__init__(path)
        self._splits: dict[int, DirType] = {}

    @property
    def splits(self) -> dict[int, DirType]:
        return self._splits

    @property
    def splits_list(self) -> list[int]:
        return self._items_list

    def create_split(
        self, split_idx: int, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(split_idx, overwrite=overwrite, exist_ok=exist_ok)


class EpochsDir(CollectionOfDirs[DirType, int]):
    _item_key = EPOCH
    _item_mapping = staticmethod(lambda x: int(x))

    def __init__(self, path: Path):
        super().__init__(path)
        self._epochs: dict[int, DirType] = {}

    @property
    def epochs(self) -> dict[int, DirType]:
        return self._epochs

    @property
    def epochs_list(self) -> list[int]:
        return self._items_list

    def create_epoch(
        self, epoch: int, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(epoch, overwrite=overwrite, exist_ok=exist_ok)


class BestModelsDir(CollectionOfDirs[DirType, str]):
    _item_key = BEST

    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics: dict[str, DirType] = {}

    @property
    def metrics(self) -> dict[str, DirType]:
        return self._metrics

    @property
    def metrics_list(self) -> list[str]:
        return self._items_list

    def create_metric(
        self, metric: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(metric, overwrite=overwrite, exist_ok=exist_ok)

    @classmethod
    def _items_dict_private_name(cls) -> str:
        return "_" + METRICS
