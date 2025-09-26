from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    CAPS,
    DATA,
    DATASET,
    SPLIT,
    TRAIN,
    VALIDATION,
)
from clinicadl.split.split import Split
from clinicadl.utils.typing import PathType

from ...base import Directory
from .splits import DataSplitDir


class DataTrainValDir(Directory):
    def __init__(self, parents_path: PathType, name: str):
        super().__init__(path=Path(parents_path) / name)

        self.splits: Dict[int, DataSplitDir] = {}

    def _create(self, dataset: CapsDataset, split: int):
        super()._create(_exists_ok=True)
        split_dir = DataSplitDir(parents_path=self.path, split=split)
        split_dir._create(dataset=dataset)
        self.splits[split] = split_dir

    def load(self):
        super().load()

        for idx in self.split_list:
            split = DataSplitDir(parents_path=self.path, split=idx)
            split.load()
            self.splits[idx] = split

    @property
    def split_list(self) -> list[int]:
        if self.is_empty():
            return []
        return sorted(
            [
                int(x.name.split("-")[1])
                for x in self.path.iterdir()
                if x.is_dir() and x.name.startswith(SPLIT)
            ]
        )


class DataDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / DATA)

        self.train = DataTrainValDir(parents_path=self.path, name=TRAIN)
        self.val = DataTrainValDir(parents_path=self.path, name=VALIDATION)

        self.df = None

    def _create(self, split: Split):
        super()._create(_exists_ok=True)
        self.train._create(dataset=split.train_dataset, split=split.index)
        self.val._create(dataset=split.val_dataset, split=split.index)

    def load(self):
        super().load()
        self.train.load()
        self.val.load()
        self.df = pd.read_csv(self.data_tsv, sep="\t")

    # def get_caps_dataset(self):
    #     return CapsDataset.from_json(self.caps_dataset_json)

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)
