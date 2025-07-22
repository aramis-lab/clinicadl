from abc import abstractmethod
from pathlib import Path

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    DATA,
    MAPS,
    SPLIT,
)
from clinicadl.tsvtools.utils import df_to_tsv
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ...maps.base import Directory


class BaseDataGroup(Directory):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)

    @property
    def maps_json(self) -> Path:
        return self.path / (MAPS + JSON)

    def create(self, dataset: CapsDataset) -> None:
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Data group '{self.name}' already exists."
            )

        self.path.mkdir(parents=True)
        df_to_tsv(tsv_path=self.data_tsv, df=dataset.df)
        # self._write_maps_json(dataset) TODO: check what's in the maps.json here


class DataGroup(BaseDataGroup):
    def __init__(self, name: str, parent_dir: PathType):
        self.name_ = name

        super().__init__(path=Path(parent_dir) / name)

    @property
    def name(self) -> str:
        return self.name_


class TrainValDataGroup(BaseDataGroup):
    def __init__(self, name: str, parent_dir: PathType, split: int):
        self.name_ = name
        self.split = split
        super().__init__(path=Path(parent_dir) / name / f"{SPLIT}-{split}")

    @property
    def name(self) -> str:
        return self.name_
