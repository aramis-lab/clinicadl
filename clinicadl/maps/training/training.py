from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.callbacks.factory.base import Callback
from clinicadl.callbacks.handler import CallbacksHandler
from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    CALLBACKS,
    COMPUTATIONAL,
    METRICS,
    OPTIMIZATION,
    TRAINING,
)
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.optim.config import OptimizationConfig
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.typing import PathType

from ..base import Directory
from .data import DataDir
from .split import TrainSplitDir


class TrainingDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / TRAINING)

        self.data = DataDir(parents_path=self.path)
        self.splits: Dict[int, TrainSplitDir] = {}

    def create_split(self, num: int):
        split = TrainSplitDir(num=num, parents_path=self.path)
        split.create()
        self.splits[num] = split

    @property
    def split_list(self) -> list[int]:
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith("split")
        ]

    def load(self):
        super().load()
        self.data.load()

        for idx in self.split_list:
            split = TrainSplitDir(num=idx, parents_path=self.path)
            split.load()
            self.splits[idx] = split

    def get_callbacks(self) -> list[Callback]:
        return CallbacksHandler.from_json(self.callbacks_json)

    def get_metrics(self) -> Dict[str, MetricConfig]:
        return MetricsHandler.from_json(self.metrics_json)

    def get_computational_config(self) -> ComputationalConfig:
        return ComputationalConfig.from_json(self.computational_json)

    def get_optimization_config(self) -> OptimizationConfig:
        return OptimizationConfig.from_json(self.optimization_json)

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
