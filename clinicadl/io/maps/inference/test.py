from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.words import METRICS

from ...base import Directory
from ..metrics import MetricsDir
from .base import (
    InferenceDir,
    InferenceGroupDir,
    InferenceResultsDir,
    InferenceSplitDir,
)


class TestModelDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics = MetricsDir(path=self.path / METRICS)

    @property
    def metrics(self) -> MetricsDir:
        return self._metrics


class TestSplitDir(InferenceSplitDir[TestModelDir]):
    _dir_type = TestModelDir


class TestResultsDir(InferenceResultsDir[TestSplitDir]):
    _dir_type = TestSplitDir


class TestGroupDir(InferenceGroupDir[TestResultsDir]):
    _results_dir_type = TestResultsDir


class TestDir(InferenceDir[TestGroupDir]):
    _dir_type = TestGroupDir
