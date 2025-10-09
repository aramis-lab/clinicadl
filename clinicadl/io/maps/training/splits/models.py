from __future__ import annotations

from pathlib import Path

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import (
    BEST,
    CHECKPOINTS,
    FINAL,
    METRICS,
    MODEL,
    MODELS,
    VALIDATION,
)

from ....base import Directory
from ...metrics import MetricsDir
from ...utils import BestModelsDir, EpochsDir


class ModelDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )

    @property
    def model(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics


class BestModelsResultsDir(BestModelsDir[ModelDir]):
    _dir_type = ModelDir


class CheckpointsResultsDir(EpochsDir[ModelDir]):
    _dir_type = ModelDir


class ModelsDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._best_models = BestModelsResultsDir(path=self.path / f"{BEST}_{MODELS}")
        self._checkpoints = CheckpointsResultsDir(path=self.path / CHECKPOINTS)
        self._final = ModelDir(path=self.path / FINAL)

    @property
    def best_models(self) -> BestModelsResultsDir:
        return self._best_models

    @property
    def checkpoints(self) -> CheckpointsResultsDir:
        return self._checkpoints

    @property
    def final(self) -> ModelDir:
        return self._final
