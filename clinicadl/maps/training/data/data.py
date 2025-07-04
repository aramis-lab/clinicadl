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


class DataDir(Directory):
    def __init__(self, parents_path: PathType):
        super().__init__(path=Path(parents_path) / DATA)

        self.train = DataTrainDir(parents_path=self.path)
        self.val = DataValDir(parents_path=self.path)

        self.df = None

    def create(self, split: Split):
        super().create(_exists_ok=True)
        self.train.create(split=split.train)
        self.val.create(split=split.val)

    def load(self):
        super().load()
        self.train.load()
        self.val.load()
        self.df = pd.read_csv(self.data_tsv, sep="\t")

    def get_caps_dataset(self):
        return CapsDataset.from_json(self.caps_dataset_json)

    @property
    def caps_dataset_json(self) -> Path:
        return self.path / (CAPS + "_" + DATASET + JSON)

    @property
    def data_tsv(self) -> Path:
        return self.path / (DATA + TSV)
