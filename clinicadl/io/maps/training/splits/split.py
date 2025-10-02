from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, LOG
from clinicadl.dictionary.words import (
    CHECKPOINTS,
    COMPUTATIONAL,
    LOGS,
    METRICS,
    SUMMARY,
    TMP,
    VALIDATION,
)

from ...metrics import MetricsDir
from ...utils import BestModelsDir
from .epoch import BestEpochDir, CheckpointsDir, TmpDir
from .logs import LogsDir


class TrainingSplitDir(BestModelsDir[BestEpochDir]):
    _dir_type = BestEpochDir

    def __init__(self, path: Path):
        super().__init__(path)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )
        self._checkpoints = CheckpointsDir(path=self.path / CHECKPOINTS)
        self._logs = LogsDir(path=self.path / LOGS)
        self._tmp = TmpDir(path=self.path / TMP)

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics

    @property
    def checkpoints(self) -> CheckpointsDir:
        return self._checkpoints

    @property
    def logs(self) -> LogsDir:
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
