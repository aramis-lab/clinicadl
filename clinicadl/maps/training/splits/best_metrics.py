from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import PTH, TAR, TSV
from clinicadl.dictionary.words import (
    BEST,
    METRICS,
    MODEL,
    OPTIMIZER,
    VALIDATION,
)
from clinicadl.utils.typing import PathType

from ...base import Directory


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
    def optimizer(self) -> Path:
        return self.path / (OPTIMIZER + PTH + TAR)

    @property
    def validation_metrics_tsv(self) -> Path:
        return self.path / (VALIDATION + "_" + METRICS + TSV)
