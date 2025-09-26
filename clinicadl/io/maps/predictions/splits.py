from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    BEST,
    COMPUTATIONAL,
    SPLIT,
)
from clinicadl.utils.typing import PathType

from ..base import Directory
from .best_metric import BestMetric


class PredSplitDir(Directory):
    def __init__(self, num: int, parent_path: PathType):
        super().__init__(path=Path(parent_path) / (SPLIT + "-" + str(num)))

        self.best_metrics: Dict[str, BestMetric] = {}

    def _create(self, metric: str):
        super()._create()
        best_metric = BestMetric(parent_dir=self.path, metric=metric)
        best_metric._create()
        self.best_metrics[metric] = best_metric

    @property
    def metric_list(self) -> list[str]:
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(BEST)
        ]

    def load(self):
        super().load()
        for metric in self.metric_list:
            best_metric = BestMetric(parent_dir=self.path, metric=metric)
            best_metric.load()
            self.best_metrics[metric] = best_metric

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)
