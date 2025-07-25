from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import (
    BEST,
    CAPS,
    METRICS,
    OUTPUT,
)
from clinicadl.utils.typing import PathType

from ..base import Directory


class BestMetric(Directory):
    def __init__(self, parent_dir: PathType, metric: str):
        super().__init__(path=Path(parent_dir) / (BEST + "-" + metric))

    @property
    def caps_output(self) -> Path:
        return self.path / (CAPS + "-" + OUTPUT)

    @property
    def metrics_tsv(self) -> Path:
        return self.path / (METRICS + TSV)
