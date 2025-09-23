from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from clinicadl.dictionary.suffixes import JSON, TSV, TXT
from clinicadl.dictionary.words import (
    CAPS_DATASET,
    DATALOADER,
    METRICS,
    PERFORMANCE,
    SPLIT,
    SUMMARY,
    TORCH,
    VALIDATION,
)
from clinicadl.utils.typing import PathType

from ...base import Directory
from .best_metric import TrainBestMetricDir
from .checkpoints import CheckpointsDir
from .logs import LogsDir
from .tmp import TmpDir


class TrainSplitDir(Directory):
    def __init__(self, num: int, parents_path: PathType):
        super().__init__(path=Path(parents_path) / (SPLIT + "-" + str(num)))

        self.best_metrics: Dict[str, TrainBestMetricDir] = {}

        self.checkpoints = CheckpointsDir(parents_path=self.path)
        self.logs = LogsDir(parents_path=self.path)
        self.tmp = TmpDir(parents_path=self.path)

    def load(self):
        super().load()
        self.checkpoints.load()
        self.logs.load()
        if self.tmp.exists():
            self.tmp.load()

        for metric in self.best_metrics_list:
            best_metric = TrainBestMetricDir(parent_dir=self.path, metric=metric)
            best_metric.load()
            self.best_metrics[metric] = best_metric

    def _create(self):
        super()._create()
        self.checkpoints._create()
        self.logs._create()
        self.tmp._create()

    def _create_best_metrics(self, metric: str):
        best_metric = TrainBestMetricDir(parent_dir=self.path, metric=metric)
        best_metric._create()
        self.best_metrics[metric] = best_metric

    @property
    def best_metrics_list(self):
        if self.is_empty():
            return []
        return sorted(
            [
                x.name.split("-")[1]
                for x in self.path.iterdir()
                if x.is_dir() and x.name.startswith("best")
            ]
        )

    @property
    def dataloader_json(self) -> Path:
        return self.path / (DATALOADER + JSON)

    @property
    def caps_dataset_json(self) -> Path:
        return self.path / (CAPS_DATASET + JSON)

    @property
    def torchsummary_txt(self) -> Path:
        return self.path / (TORCH + SUMMARY + TXT)

    @property
    def validation_metrics_tsv(self) -> Path:
        return self.path / (VALIDATION + "_" + METRICS + TSV)

    @property
    def performance_txt(self) -> Path:
        return self.path / (PERFORMANCE + TXT)
