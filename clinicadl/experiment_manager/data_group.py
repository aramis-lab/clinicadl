import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import DATA, GROUPS, MAPS, SPLIT, TRAIN, VALIDATION
from clinicadl.tsvtools.utils import df_to_tsv, tsv_to_df
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError

TRAIN_VAL = [TRAIN, VALIDATION]


class SplitGroup(ClinicaDLConfig):
    number: int


class DataGroup(ClinicaDLConfig):
    maps_path: Path
    name: str
    split: Optional[int] = None

    @property
    def data_tsv(self) -> Path:
        return self.group_split_dir / (DATA + TSV)

    @property
    def maps_json(self) -> Path:
        return self.group_split_dir / (MAPS + JSON)

    @property
    def group_dir(self) -> Path:
        return self.maps_path / GROUPS / self.name

    @property
    def group_split_dir(self) -> Path:
        if self.name in TRAIN_VAL:
            if self.split is not None:
                path_ = self.group_dir / (SPLIT + "-" + str(self.split))
            else:
                raise ClinicaDLConfigurationError(
                    "You must specify a split number when the data group is 'train' or 'validation'."
                )
        else:
            path_ = self.group_dir

        path_.mkdir(parents=True, exist_ok=True)
        return path_

    @property
    def df(self) -> pd.DataFrame:
        return tsv_to_df(self.data_tsv)

    @property
    def caps_dir(self) -> Path:
        dict_ = self._read_json()
        return Path(dict_["cap_directory"])

    def exists(self) -> bool:
        return self.data_tsv.is_file() and self.maps_json.is_file()

    def create(self, caps_dataset: CapsDataset):
        if self.data_tsv.is_file():
            raise FileExistsError(
                f"The data group {self.name} already exists in {self.maps_path}"
            )
        df_to_tsv(self.data_tsv, caps_dataset.df)
        self._write_json(caps_dataset)

    def _write_json(self, caps_dataset: CapsDataset):
        dict_ = caps_dataset.describe()  # TODO: to complete
        with self.maps_json.open(mode="w") as file:
            json.dump(dict_, file)

    def _read_json(self):
        if self.maps_json.is_file():
            with self.maps_json.open(mode="r") as file:
                dict_ = json.load(file)
                return dict_

        raise FileNotFoundError(
            f"Could not find the `maps.json` file for the data grou : {self.name} (in {self.maps_path})"
        )
