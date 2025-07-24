from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    CALLBACKS,
    COMPUTATIONAL,
    METRICS,
    OPTIMIZATION,
    TRAINING,
)
from clinicadl.utils.typing import PathType

from ..base import Directory
from .data import DataDir
from .splits import TrainSplitDir


class TrainingDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / TRAINING)

        self.data = DataDir(parents_path=self.path)
        self.splits: Dict[int, TrainSplitDir] = {}

    def _create_split(self, num: int) -> None:
        split = TrainSplitDir(num=num, parents_path=self.path)
        split._create()
        self.splits[num] = split

    @property
    def split_list(self) -> list[int]:
        if self.is_empty():
            return []
        return sorted(
            [
                int(x.name.split("-")[1])
                for x in self.path.iterdir()
                if x.is_dir() and x.name.startswith("split")
            ]
        )

    def load(self):
        super().load()
        self.data.load()

        for idx in self.split_list:
            split = TrainSplitDir(num=idx, parents_path=self.path)
            split.load()
            self.splits[idx] = split

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def optimization_json(self) -> Path:
        return self.path / (OPTIMIZATION + JSON)

    @property
    def callbacks_json(self) -> Path:
        return self.path / (CALLBACKS + JSON)

    @property
    def metrics_json(self) -> Path:
        return self.path / (METRICS + JSON)
