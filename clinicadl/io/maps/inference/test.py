from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.words import METRICS

from ...base import Directory
from ..metrics import MetricsDir
from .base import InferenceDir, InferenceGroupDir, InferenceSplitDir


class ModelDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics = MetricsDir(path=self.path / METRICS)

    @property
    def metrics(self) -> MetricsDir:
        return self._metrics


class TestSplitDir(InferenceSplitDir[ModelDir]):
    _dir_type = ModelDir


class TestGroupDir(InferenceGroupDir[TestSplitDir]):
    _dir_type = TestSplitDir


class TestDir(InferenceDir[TestGroupDir]):
    _dir_type = TestGroupDir
