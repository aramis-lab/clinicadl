from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, TXT
from clinicadl.dictionary.words import (
    ARCHITECTURE,
    ENVIRONMENT,
    MODEL,
    SUMMARY,
    TIME,
)
from clinicadl.models import ClinicaDLModel
from clinicadl.split.split import Split
from clinicadl.utils.typing import PathType

from .base import Directory
from .predictions import PredictionsDir
from .training import TrainingDir


class Maps(Directory):
    """
    Entry point to access a ClinicaDL MAPS directory.

    The ``Maps`` class provides access to the structure and content of a MAPS
    directory, including training data, prediction results, model checkpoints,
    and associated metadata.

    Typically, users only need to call the ``load()`` method to inspect or reuse
    an already trained model directory.

    Attributes
    ----------
    training : :py:class:`~clinicadl.IO.maps.training.TrainingDir`
        Access to training-related files (data, splits, checkpoints).
    predictions : :py:class:`~clinicadl.IO.maps.predictions.PredictionsDir`
        Access to prediction results for test groups.

    Examples
    --------
    .. code-block:: python

        from clinicadl.IO.maps import Maps
        maps = Maps("/path/to/maps_dir")
        maps.load()  # Load existing structure

        maps.training.split_list
        >>> [0, 1, 2, 3, 4]

        first_split = maps.training.splits[0]
        print(first_split.logs.training_tsv)
        >>> /path/to/maps_dir/training/split-0/logs/training.tsv

        pred_group = maps.predictions.groups["CNvsAD"]
        metrics_file = pred_group.splits[0].best_metrics["loss"].metrics_tsv
        print(metrics_file)
        >>> /path/to/maps_dir/predictions/testCNvsAD/split-0/best-loss/metrics.tsv
    """

    def __init__(self, maps_path: PathType):
        """
        Initialize a Maps object with the given path.

        This does not read or create any directories — call `load()` to populate the structure.
        """
        super().__init__(path=maps_path)

        self.predictions = PredictionsDir(parents_path=self.path)
        self.training = TrainingDir(parents_path=self.path)

    def create(self, overwrite: bool = False) -> None:
        """
        Create the directory if it does not already exist or if overwrite is True.

        Also create the training and prediction directories and write the environment.txt file.

        """
        super()._create(overwrite=overwrite)
        self.predictions._create(overwrite=overwrite)
        self.training._create(overwrite=overwrite)
        self._write_environment_txt()
        self._create_summary_log()

    def _create_training_split(self, split: Split) -> None:
        """
        Create a new split in the training directory:
         - Create a new split directory
         - Create the data.tsv files in the data directory of the new split

        """
        self.training._create_split(num=split.index)
        self.training.data._create(split=split)

    def load(self) -> None:
        """
        Load the MAPS directory structure from disk.

        This method reads all subfolders (training, predictions, data splits, etc.)
        and reconstructs the directory tree as Python objects.

        After loading, you can navigate through the :class:`~clinicadl.IO.maps.Maps` object to access all components.

        Directory Layout
        ----------------
        Example structure after training:

        .. code-block:: text

            maps_path/
            ├── architecture.log
            ├── environment.txt
            ├── model.json
            ├── summary.log
            ├── predictions
            │   └── test<GroupName>
            │       ├── data.tsv
            │       ├── metrics.json
            │       ├── caps_dataset.json
            │       └── split-<N>
            │           ├── best-<metric>
            │           │   ├── metrics.tsv
            │           │   └── caps_output/
            │           └── computational.json
            └── training
                ├── data
                │   ├── data.tsv
                │   ├── caps_dataset.json
                │   ├── train
                │   │   └── split-<N>
                │   │       └── data.tsv
                │   └── validation
                │       └── split-<N>
                │           └── data.tsv
                ├── split-<N>
                │   ├── best-<metric>
                │   │   ├── model.pth.tar
                │   │   └── optimizer.pth.tar
                │   ├── checkpoints
                │   │   └── epoch-<K>
                │   │       ├── model.pth.tar
                │   │       └── optimizer.pth.tar
                │   ├── logs
                │   │   └── training.tsv
                │   └── tmp
                │       ├── model.pth.tar
                │       └── optimizer.pth.tar
                ├── computational.json
                ├── optimization.json
                ├── metrics.json
                └── callbacks.json

        .. note::
            - ``<N>`` refers to the split index (e.g., 0, 1, 2).
            - ``<GroupName>`` refers to the name of the test group (e.g., ``CNvsAD``).
            - ``<metric>`` refers to the metric used to select the best model (e.g., ``loss``).

        """

        super().load()

        self.predictions.load()
        self.training.load()

    @property
    def architecture_log(self) -> Path:
        return self.path / (ARCHITECTURE + LOG)

    @property
    def model_json(self) -> Path:
        return self.path / (MODEL + JSON)

    @property
    def environment_txt(self) -> Path:
        return self.path / (ENVIRONMENT + TXT)

    @property
    def summary_log(self) -> Path:
        return self.path / (SUMMARY + LOG)

    def _create_summary_log(self):
        """Create a summary log file."""

        summary = "==================== Summary Log ===================="
        summary += "\n\n"
        summary += (
            f"Date              : {datetime.now().strftime('%d %b %Y, %H:%M:%S')}"
        )
        summary += "\n"
        summary += f"Path              : {self.path.resolve()}"
        summary += "\n"

        with (self.summary_log).open(mode="w") as file:
            file.write(summary)

    def _add_lines_to_summary_log(self, line):
        line += "\n"
        with (self.summary_log).open(mode="a") as file:
            file.write(line)

    def _write_environment_txt(self) -> None:
        """Writes the installed Python packages (via `pip freeze`) to `environment.txt`."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.environment_txt).open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            with (self.environment_txt).open(mode="w") as file:
                file.write("pip freeze")
