from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.dictionary.words import (
    CAPS,
    DATA,
    DATASET,
    METRICS,
    SPLIT,
    TEST,
)
from clinicadl.utils.typing import PathType

from ..base import Directory
from .splits import PredSplitDir


class GroupDir(Directory):
    def __init__(self, parents_path: PathType, group_name: str):
        super().__init__(path=Path(parents_path) / (TEST + "-" + group_name))

        self.splits: Dict[int, PredSplitDir] = {}

    def _create(self, split: int, metric: str, dataset: CapsDataset):
        super()._create()

        dataset.df.to_csv(self.data_tsv, sep="\t", index=False)

        split_dir = PredSplitDir(num=split, parent_path=self.path)
        split_dir._create(metric=metric)
        self.splits[split] = split_dir

    def load(self):
        super().load()
        for idx in self.split_list:
            split = PredSplitDir(num=idx, parent_path=self.path)
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

    @property
    def caps_dataset_json(self) -> Path:
        return self.path / (CAPS + "_" + DATASET + JSON)

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)

    @property
    def metrics_json(self) -> Path:
        return self.path / (METRICS + JSON)
