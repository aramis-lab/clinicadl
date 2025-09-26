from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    BEST,
    MODEL,
)
from clinicadl.utils.typing import PathType

from ...base import Directory
from .metrics import MetricsDir


class TrainBestMetricDir(Directory):
    def __init__(self, parent_dir: PathType, metric: str):
        super().__init__(path=Path(parent_dir) / (BEST + "-" + metric))

    def load(self):
        super().load()
        # TODO: some check ?

    @property
    def model(self) -> Path:
        return self.path / (MODEL + PTH + TAR)

    @property
    def metrics(self) -> MetricsDir:
        return MetricsDir(parent_dir=self.path)
