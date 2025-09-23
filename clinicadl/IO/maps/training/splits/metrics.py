from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import METRICS
from clinicadl.utils.typing import PathType

from ...base import Directory


class MetricsDir(Directory):
    def __init__(self, parent_dir: PathType):
        super().__init__(path=Path(parent_dir) / METRICS)

    @property
    def validation(self) -> Path:
        return (self.path / "validation").with_suffix(TSV)

    @property
    def validation_details(self) -> Path:
        return (self.path / "validation_details").with_suffix(TSV)
