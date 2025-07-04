from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from clinicadl.callbacks.factory.base import Callback
from clinicadl.callbacks.handler import CallbacksHandler
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.dictionary.words import (
    ARCHITECTURE,
    BEST,
    CALLBACKS,
    CAPS,
    CHECKPOINTS,
    COMPUTATIONAL,
    DATA,
    DATASET,
    ENVIRONMENT,
    EPOCH,
    GROUPS,
    LOGS,
    METRICS,
    MODEL,
    OPTIMIZATION,
    OPTIMIZER,
    OUTPUT,
    PREDICTIONS,
    SPLIT,
    SUMMARY,
    TEST,
    TMP,
    TRAIN,
    TRAINING,
    VALIDATION,
)
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.tsvtools.utils import remove_non_empty_dir
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ...base import Directory


class TrainSplitDir(Directory):
    def __init__(self, num: int, parents_path: PathType):
        super().__init__(path=Path(parents_path) / (SPLIT + "-" + str(num)))

        self.best_metrics: Dict[str, TrainBestMetric] = {}

        self.checkpoints = CheckpointsDir(parents_path=self.path)
        self.logs = LogsDir(parents_path=self.path)

        self.tmp = TmpDir(parent_dir=self.path)

    def load(self):
        super().load()
        self.checkpoints.load()
        self.logs.load()
        self.tmp.load()

        for metric in self.best_metrics_list:
            best_metric = TrainBestMetric(parent_dir=self.path, metric=metric)
            best_metric.load()
            self.best_metrics[metric] = best_metric

    def get_computational_config(self) -> ComputationalConfig:
        return ComputationalConfig.from_json(self.computational_json)

    @property
    def best_metrics_list(self):
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith("best")
        ]

    @property
    def computational_json(self) -> Path:
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def summary_log(self) -> Path:
        return self.path / (SUMMARY + LOG)

    @property
    def validation_metrics_tsv(self) -> Path:
        return self.path / (VALIDATION + "_" + METRICS + TSV)
