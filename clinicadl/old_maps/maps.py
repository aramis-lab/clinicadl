from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Union

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.dictionary.words import (
    COMPUTATIONAL,
    ENVIRONMENT,
    GROUPS,
    METRICS,
    MODEL,
    OPTIMIZATION,
    TRAIN,
    VALIDATION,
)
from clinicadl.split.split import Split
from clinicadl.tsvtools.utils import remove_non_empty_dir
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ..maps.base import Directory
from .data_group import DataGroup, TrainValDataGroup
from .split_dir import SplitDir

TRAIN_VAL = [TRAIN, VALIDATION]
DataGroupType = Union[DataGroup, TrainValDataGroup]


class Maps(Directory):
    """
    Class representing the `MAPS` (Model Analysis and Processing Structure) folder.
    This directory contains all elements obtained during training, validation, and post-processing
    procedures in a deep learning framework.

    The structure is organized into:
    - **Splits**: A training procedure consists of training one model per train/validation split
      defined by the validation procedure. The MAPS directory contains `split-<i>` folders, where `i`
      ranges from `0` to `N-1`, each storing information about the corresponding split.
    - **Best Metrics**: For each split, a model is selected per user-defined `selection_metrics`.
      The output folder for a model selected by a metric `<metric>` is named `best-<metric>`.
    - **Data Groups**: A selected model can be applied to different datasets for individual predictions,
      evaluation metrics, or interpretability maps. These datasets, called "data groups," are stored
      at the root of the MAPS directory to ensure their characteristics are shared across all models.

    Attributes
    ----------
    path : Path
        Path to the MAPS directory.
    splits : Dict[int, SplitDir]
        Dictionary mapping split indices to their corresponding `SplitDir` objects.
    data_groups : Dict[str, Union[Dict[int, DataGroupType], DataGroupType]]
        Dictionary storing data groups. For train/validation data groups, a dictionary is maintained
        per split. For other data groups, they are stored individually.
    """

    def __init__(self, maps_path: PathType, overwrite: bool = False):
        super().__init__(path=maps_path)

        self._overwrite = overwrite
        self.splits: Dict[int, SplitDir] = {}
        self.data_groups: Dict[str, Union[Dict[int, DataGroupType], DataGroupType]] = {}

    def load(self):
        """
        Loads an existing MAPS directory.

        Parameters
        ----------
        maps_path : PathType
            Path to the MAPS directory.

        Returns
        -------
        Maps
            An instance of the `Maps` class representing the loaded directory.
        """
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")

        for split_idx in self.split_list:
            split_dir = SplitDir.load(num=split_idx, maps_path=self.path)

            if not split_dir.exists() or split_dir.is_empty():
                raise ClinicaDLConfigurationError(
                    f"The split at {split_dir.path} doesn't exist or is empty."
                )

            self.splits[split_idx] = split_dir

        for data_group in self.group_list:
            if data_group in TRAIN_VAL:
                for split_idx in self.splits.keys():
                    group = TrainValDataGroup(
                        name=data_group, parent_dir=self.groups_dir, split=split_idx
                    )

                    self.data_groups[group.name] = {split_idx: group}

            else:
                group = DataGroup(name=data_group, parent_dir=self.groups_dir)
                if not group.exists():
                    raise ClinicaDLConfigurationError(
                        f"The group at {group.path} doesn't exist."
                    )
                self.data_groups[group.name] = group

    @property
    def groups_dir(self) -> Path:
        """Returns the path to the groups directory inside MAPS."""
        return self.path / GROUPS

    @property
    def json_dir(self) -> Path:
        return self.path / "json"

    @property
    def split_list(self) -> list[int]:
        """Returns a list of available split indices."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []
        return [
            int(x.name.split("-")[1])
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith("split")
        ]

    @property
    def group_list(self) -> list[str]:
        """Returns a list of available data group names."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []
        return [x.name for x in self.groups_dir.iterdir() if x.is_dir()]

    @property
    def train_val_tsv(self) -> Path:
        """Returns the path to the `train+validation.tsv` file."""
        return (self.path / f"{TRAIN}+{VALIDATION}").with_suffix(TSV)

    @property
    def requirements_txt(self) -> Path:
        """Returns the path to the `environment.txt`file."""
        return (self.path / ENVIRONMENT).with_suffix(TXT)

    @property
    def metrics_json(self) -> Path:
        """Returns the path to the `maps.json` configuration file."""
        return (self.json_dir / METRICS).with_suffix(JSON)

    @property
    def model_json(self) -> Path:
        """Returns the path to the `model.json` configuration file."""
        return (self.json_dir / MODEL).with_suffix(JSON)

    @property
    def computational_json(self) -> Path:
        """Returns the path to the `computational.json` configuration file."""
        return (self.json_dir / COMPUTATIONAL).with_suffix(JSON)

    @property
    def optimization_json(self) -> Path:
        """Returns the path to the `optimization.json` configuration file."""
        return (self.json_dir / OPTIMIZATION).with_suffix(JSON)

    def create_data_group(self, name: str, dataset: CapsDataset) -> None:
        """
        Creates a new data group within the MAPS directory.

        Parameters
        ----------
            name: str
                Name of the data group.
            dataset: CapsDataset
                Dataset associated with the data group.

        Raises
        ------
            ClinicaDLConfigurationError: If the data group already exists.
        """
        if name in self.data_groups:
            raise ClinicaDLConfigurationError(f"Data group '{name}' already exists.")

        data_group = DataGroup(name=name, parent_dir=self.groups_dir)
        data_group._create(dataset=dataset)
        self.data_groups[name] = data_group

    def create_split(self, split: Split, best_metrics: list[str]) -> None:
        """
        Creates a new split directory within the MAPS directory.
        Creates the train and validation data_group associated to this split.

        Parameters
        ----------
            split: Split
                Split object defining train/validation datasets.
            best_metrics: list[str]
                List of metrics used for model selection.

        Raises
        ------
            ClinicaDLConfigurationError: If the split already exists.
        """
        if split.index in self.splits:
            raise ClinicaDLConfigurationError(f"Split '{split.index}' already exists.")

        split_dir = SplitDir(
            num=split.index, best_metrics=best_metrics, maps_path=self.path
        )
        split_dir.create(split=split)
        self.splits[split.index] = split_dir

        train_group = TrainValDataGroup(
            name=TRAIN, parent_dir=self.groups_dir, split=split.index
        )
        train_group.create(dataset=split.train_dataset)
        self.data_groups[TRAIN] = {split.index: train_group}

        val_group = TrainValDataGroup(
            name=VALIDATION, parent_dir=self.groups_dir, split=split.index
        )
        val_group.create(dataset=split.val_dataset)
        self.data_groups[VALIDATION] = {split.index: val_group}

    def create(self) -> None:
        """
        Creates the MAPS directory if it does not already exist.

        Raises:
            ClinicaDLConfigurationError: If the directory already exists.
        """
        if self.exists() and not self._overwrite:
            raise ClinicaDLConfigurationError(
                f"Maps directory ({self.path})already exists."
            )
        elif self._overwrite and self.exists():
            self.remove()

        self.path.mkdir(parents=True, exist_ok=True)
        self.groups_dir.mkdir(parents=True)
        self.json_dir.mkdir(parents=True)
        self._write_requirements_version()

    def _write_requirements_version(self) -> None:
        """Writes the installed Python packages (via `pip freeze`) to `environment.txt`."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.requirements_txt).open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            with (self.requirements_txt).open(mode="w") as file:
                file.write("pip freeze")

    def read_json(self) -> dict:
        return dict()

    def caps_dir(self) -> Path:  # TODO: to change !
        return self.read_json().get("caps_dir", Path(""))

    def remove(self) -> None:
        remove_non_empty_dir(self.path)
