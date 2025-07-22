from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from clinicadl.dictionary.suffixes import JSON, PTH, TAR, TSV
from clinicadl.dictionary.words import (
    BEST,
    CHECKPOINT,
    EPOCH,
    METRICS,
    MODEL,
    OPTIMIZER,
    SPLIT,
    TMP,
    TRAINING,
)
from clinicadl.split.split import Split
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ...maps.base import Directory
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

        self.epochs: Dict[int, EpochDir] = {}

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

        # for epoch in split_dir.epoch_dir_list:
        #     epoch_dir = EpochDir.load(parent_dir=split_dir.path, epoch=epoch)
        #     if not epoch_dir.exists() or epoch_dir.is_empty():
        #         raise ClinicaDLConfigurationError(
        #             f"The epoch at {epoch_dir.path} doesn't exist or is empty."
        #         )
        #     split_dir.epochs[epoch] = epoch_dir

        # TODO : load tmp et training ?
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
    def epoch_dir_list(self) -> list[str]:
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []

        return [
            x.name
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(EPOCH)
        ]

    @property
    def split_json(self) -> Path:
        """Returns the path to the `split.json` file storing the split configuration."""
        return (self.path / SPLIT).with_suffix(JSON)

    @property
    def metrics_tsv(self) -> Path:
        return (self.path / METRICS).with_suffix(TSV)

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
            raise ClinicaDLConfigurationError(
                f"Split '{self.number}' already exists at {self.path}."
            )
        self.path.mkdir(parents=True)
        for metric in self.best_metrics.values():
            metric.create(split=split)

    def create_epoch(self, epoch: int):
        epoch_dir = EpochDir(parent_dir=self.path, epoch=epoch)
        epoch_dir.path.mkdir(parents=True)
        self.epochs[epoch] = epoch_dir

    def plot_loss(self):
        """Plots the training loss over time using the data from the training log file."""
        self.logs.plot_loss()


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

    def plot_loss(self):
        if not self.training_tsv.is_file():
            raise ClinicaDLConfigurationError(
                f"The training log file at {self.training_tsv} doesn't exist."
            )
        df = pd.read_csv(self.training_tsv, sep="\t")
        plt.figure(figsize=(12, 6))
        plt.plot(df["time"], df["loss"], label="Loss", color="blue", linewidth=2)

        # Ajouter les epochs en fond (vertical lines)
        epoch_changes = df[df["batch"] == 0]
        for _, row in epoch_changes.iterrows():
            plt.axvline(x=row["time"], color="gray", linestyle="--", alpha=0.3)
            plt.text(
                row["time"],
                max(df["loss"]),
                f"Epoch {int(row['epoch'])}",
                rotation=90,
                verticalalignment="top",
                fontsize=8,
                color="gray",
            )

        # Annotations optionnelles des batchs (plus denses)
        for i in range(
            0, len(df), 10
        ):  # afficher 1 batch sur 10 pour éviter la surcharge
            row = df.iloc[i]
            plt.text(
                row["time"],
                row["loss"],
                f"B{int(row['batch'])}",
                fontsize=6,
                alpha=0.6,
                rotation=45,
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Loss")
        plt.title("Loss(Time)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return (self.path / OPTIMIZER).with_suffix(PTH + TAR)

    def remove(self) -> None:
        """Removes the temporary files."""
        if self.checkpoint.is_file():
            self.checkpoint.unlink()
        if self.optimizer.is_file():
            self.optimizer.unlink()
        if self.path.is_dir():
            self.path.rmdir()


class EpochDir(Directory):
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

    def __init__(self, parent_dir: PathType, epoch: int):
        super().__init__(path=Path(parent_dir) / f"{EPOCH}-{epoch}")

    @property
    def checkpoint(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def optimizer(self) -> Path:
        return (self.path / OPTIMIZER).with_suffix(PTH + TAR)
