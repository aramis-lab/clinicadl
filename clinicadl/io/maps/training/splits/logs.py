from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.dictionary.words import (
    COMPUTATIONAL,
    LEARNING_RATES,
    LOSS,
    TRAINING,
)

from ....base import Directory


class TrainingLogsDir(Directory):
    @property
    def training_loss_tsv(self) -> Path:
        return (self.path / f"{TRAINING}_{LOSS}").with_suffix(TSV)

    @property
    def computational_tsv(self) -> Path:
        return (self.path / COMPUTATIONAL).with_suffix(TSV)

    @property
    def learning_rates(self) -> Path:
        return self.path / LEARNING_RATES
