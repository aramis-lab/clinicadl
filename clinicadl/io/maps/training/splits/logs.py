from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import (
    LEARNING_RATES,
    LOSS,
    TRAINING,
)

from ....base import Directory


class LogsDir(Directory):
    @property
    def training_loss(self) -> Path:
        return (self.path / f"{TRAINING}_{LOSS}").with_suffix(TSV)

    @property
    def learning_rates(self) -> Path:
        return self.path / LEARNING_RATES
