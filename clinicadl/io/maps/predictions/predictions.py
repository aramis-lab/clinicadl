from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.words import (
    PREDICTIONS,
    TEST,
)
from clinicadl.utils.typing import PathType

from ..base import Directory
from .groups import GroupDir


class PredictionsDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / PREDICTIONS)

        self.groups: Dict[str, GroupDir] = {}

    def load(self):
        super().load()
        for name in self.group_list:
            group = GroupDir(parents_path=self.path, group_name=name)
            group.load()
            self.groups[name] = group

    def _create_group(
        self, group_name: str, split: int, metric: str, dataset: CapsDataset
    ):
        super()._create(_exists_ok=True)
        group = GroupDir(parents_path=self.path, group_name=group_name)
        group._create(split=split, metric=metric, dataset=dataset)
        self.groups[group_name] = group

    @property
    def group_list(self) -> list[str]:
        if self.is_empty():
            return []
        return sorted(
            [
                x.name.split("-")[1]
                for x in self.path.iterdir()
                if x.is_dir() and x.name.startswith(TEST)
            ]
        )
