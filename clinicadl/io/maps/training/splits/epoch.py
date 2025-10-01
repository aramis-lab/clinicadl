from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, PTH, TAR
from clinicadl.dictionary.words import (
    CALLBACKS,
    METRICS,
    MODEL,
    STOP,
    VALIDATION,
)

from ...base import Directory
from ...metrics import MetricsDir
from ...utils import EpochsDir


class EpochDir(Directory):
    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)


class BestEpochDir(EpochDir):
    def __init__(self, path: Path):
        super().__init__(path)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics


class CallbacksDir(Directory):
    pass


class EpochTmpDir(BestEpochDir):
    def __init__(self, path: Path):
        super().__init__(path)
        self._callbacks = CallbacksDir(path=self.path / CALLBACKS)

    @property
    def callbacks(self) -> CallbacksDir:
        return self._callbacks

    @property
    def stop(self) -> Path:
        return (self.path / STOP).with_suffix(JSON)


class CheckpointsDir(EpochsDir[EpochDir]):
    _dir_type = EpochDir


class TmpDir(EpochsDir[EpochTmpDir]):
    _dir_type = EpochTmpDir
