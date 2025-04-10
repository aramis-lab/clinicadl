from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from clinicadl.dictionary.suffixes import JSON, PTH, TAR, TSV
from clinicadl.dictionary.words import (
    BEST,
    CHECKPOINT,
    OPTIMIZER,
    SPLIT,
    TMP,
    TRAINING,
)
from clinicadl.metrics.config import MetricConfig
from clinicadl.splitter.split import Split
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ..base import Directory
from .best_metric import BestMetric


class SplitDir(Directory):
    """Handles the structure and operations related to a specific split.

    A `SplitDir` directory contains:
    - Training logs
    - Temporary files (checkpoints, optimizers)
    - Best models selected based on various metrics
    - A JSON file storing the split configuration.

    Attributes
    ----------
        number: int
            The index of the split.
        logs: TrainingLogs
            Directory for storing training logs.
        tmp: TmpDir
            Directory for storing temporary files.
        best_metric
            Dict[str, BestMetric]): Dictionary of best models per metric.
    """

    def __init__(self, num: int, maps_path: PathType, best_metrics: list[str] = []):
        self.number = num
        super().__init__(path=Path(maps_path) / (SPLIT + "-" + str(num)))

        self.logs = TrainingLogs(parent_dir=self.path)
        self.tmp = TmpDir(parent_dir=self.path)

        self.best_metrics: Dict[str, BestMetric] = {}
        for metric in best_metrics:
            self.best_metrics[metric] = BestMetric(metric=metric, parent_dir=self.path)

        # TODO: add somethin to write split info

    @classmethod
    def load(cls, num: int, maps_path: PathType) -> SplitDir:
        """Loads an existing split directory.

        Parameters
        ----------
            num: int
                The index of the split.
            maps_path: PathType
                Path to the MAPS directory.

        Returns
        -------
            SplitDir
                An instance of the SplitDir class.
        """

        split_dir = cls(num=num, maps_path=maps_path)
        for metric in split_dir.best_metrics_list:
            best_metric = BestMetric.load(parent_dir=split_dir.path, metric=metric)

            if not best_metric.exists() or best_metric.is_empty():
                raise ClinicaDLConfigurationError(
                    f"The metric at {best_metric.path} doesn't exist or is empty."
                )

            if not split_dir.logs.exists():
                raise ClinicaDLConfigurationError(
                    f"The logs folder at {split_dir.logs.path} doesn't exist."
                )

            split_dir.best_metrics[best_metric.metric] = best_metric

        return split_dir

    @property
    def best_metrics_list(self) -> list[str]:
        """Returns a list of available metrics in the split directory."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []

        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(BEST)
        ]

    @property
    def split_json(self) -> Path:
        """Returns the path to the `split.json` file storing the split configuration."""
        return (self.path / SPLIT).with_suffix(JSON)

    def create(self, split: Split) -> None:
        """Creates the directory structure for the split and initializes required files.

        Parameters
        ----------
            split: Split
                The split object defining train/validation datasets.

        Raises
        ------
            ClinicaDLConfigurationError: If the split directory already exists.
        """
        if self.exists():
            raise ClinicaDLConfigurationError(f"Split '{self.number}' already exists.")
        self.path.mkdir(parents=True)
        for metric in self.best_metrics.values():
            metric.create(split=split)


class TrainingLogs(Directory):
    """Handles training logs for a given split.

    A `TrainingLogs` directory contains:
    - TensorBoard logs for tracking training progress.
    - A TSV file summarizing training metrics.

    Attributes
    ----------
        tensorboard: Path
            Path to the TensorBoard logs directory.
        training_tsv: Path
            Path to the training log file.
    """

    def __init__(self, parent_dir: PathType):
        super().__init__(path=Path(parent_dir) / "training_logs")

    @property
    def tensorboard(self) -> Path:
        return self.path / "tensorboard"

    @property
    def training_tsv(self) -> Path:
        return (self.path / TRAINING).with_suffix(TSV)


class TmpDir(Directory):
    """Handles temporary files related to model training.

    A `TmpDir` directory contains:
    - Model checkpoints (weights).
    - Optimizer state files.

    Attributes
    ----------
        checkpoint: Path
            Path to the checkpoint file.
        optimizer: Path
            Path to the optimizer state file.
    """

    def __init__(self, parent_dir: PathType):
        super().__init__(path=Path(parent_dir) / TMP)

    @property
    def checkpoint(self) -> Path:
        return (self.path / CHECKPOINT).with_suffix(PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return (self.path / OPTIMIZER).with_suffix(PTH + TAR)
