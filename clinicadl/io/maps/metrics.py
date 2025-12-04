from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import AGGREGATED, DETAILS

from ..base import Directory


class MetricsDir(Directory):
    @property
    def aggregated(self) -> Path:
        return (self.path / AGGREGATED).with_suffix(TSV)

    @property
    def details(self) -> Path:
        return (self.path / DETAILS).with_suffix(TSV)
