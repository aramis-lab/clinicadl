from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import JSON, LOG
from clinicadl.utils.dictionary.words import (
    COMPUTATIONAL,
    LOGS,
    METRICS,
    MODELS,
    SUMMARY,
    TMP,
    VALIDATION,
)

from ....base import Directory
from ...metrics import MetricsDir
from .logs import TrainingLogsDir
from .models import ModelsDir
from .tmp import TmpDir


class TrainingSplitDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._models = ModelsDir(path=self.path / MODELS)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )
        self._logs = TrainingLogsDir(path=self.path / LOGS)
        self._tmp = TmpDir(path=self.path / TMP)

    @property
    def models(self) -> ModelsDir:
        return self._models

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics

    @property
    def logs(self) -> TrainingLogsDir:
        return self._logs

    @property
    def tmp(self) -> TmpDir:
        return self._tmp

    @property
    def computational_json(self) -> Path:
        return (self.path / COMPUTATIONAL).with_suffix(JSON)

    @property
    def summary_log(self) -> Path:
        return (self.path / SUMMARY).with_suffix(LOG)
