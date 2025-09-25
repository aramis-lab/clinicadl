from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    CALLBACKS,
    TMP,
)
from clinicadl.utils.typing import PathType

from .checkpoints import CheckpointsDir, EpochDir
from .metrics import MetricsDir


class EpochTmpDir(EpochDir):
    @property
    def callbacks(self) -> Path:
        if not (self.path / CALLBACKS).is_dir():
            (self.path / CALLBACKS).mkdir(parents=True)
        return self.path / CALLBACKS

    @property
    def metrics(self) -> MetricsDir:
        return MetricsDir(parent_dir=self.path)

    @property
    def stop(self) -> Path:
        return (self.path / "stop").with_suffix(JSON)


class TmpDir(CheckpointsDir):
    _epoch_dir_type = EpochTmpDir

    def __init__(self, parents_path: PathType):
        super(CheckpointsDir, self).__init__(path=Path(parents_path) / TMP)

        self.epochs: dict[int, EpochTmpDir] = {}
