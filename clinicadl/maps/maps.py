import json
import subprocess
from pathlib import Path
from typing import Dict, Union

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.dictionary.words import (
    COMPUTATIONAL,
    GROUPS,
    MAPS,
    MODEL,
    OPTIMIZATION,
    TRAIN,
    VALIDATION,
)
from clinicadl.metrics.metrics import MetricConfig
from clinicadl.splitter.split import Split
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from .base import Directory
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

    def __init__(self, maps_path: PathType):
        super().__init__(path=maps_path)

        self.splits: Dict[int, SplitDir] = {}
        self.data_groups: Dict[str, Union[Dict[int, DataGroupType], DataGroupType]] = {}

    @property
    def groups_dir(self) -> Path:
        """Returns the path to the groups directory inside MAPS."""
        return self.path / GROUPS

    @property
    def split_list(self) -> list[int]:
        """Returns a list of available split indices."""
        return [int(x.name.split("-")[1]) for x in self.path.iterdir() if x.is_dir()]

    @property
    def group_list(self) -> list[str]:
        """Returns a list of available data group names."""
        return [x.name for x in self.groups_dir.iterdir() if x.is_dir()]

    @property
    def train_val_tsv(self) -> Path:
        """Returns the path to the `train+validation.tsv` file."""
        return self.path / f"{TRAIN}+{VALIDATION}{TSV}"

    @property
    def maps_json(self) -> Path:
        """Returns the path to the `maps.json` configuration file."""
        return self.path / (MAPS + JSON)

    @property
    def model_json(self) -> Path:
        """Returns the path to the `model.json` configuration file."""
        return self.path / (MODEL + JSON)

    @property
    def computational_json(self) -> Path:
        """Returns the path to the `computational.json` configuration file."""
        return self.path / (COMPUTATIONAL + JSON)

    @property
    def optimization_json(self) -> Path:
        """Returns the path to the `optimization.json` configuration file."""
        return self.path / (OPTIMIZATION + JSON)

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
        data_group.create(dataset=dataset)
        self.data_groups[name] = data_group

    def create_split(self, split: Split, best_metrics: list[MetricConfig]) -> None:
        """
        Creates a new split directory within the MAPS directory.
        Creates the train and validation data_group associated to this split.

        Parameters
        ----------
            split: Split
                Split object defining train/validation datasets.
            best_metrics: list[MetricConfig]
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
            name=TRAIN, parent_dir=self.path, split=split.index
        )
        train_group.create(dataset=split.train_dataset)
        self.data_groups[TRAIN] = {split.index: train_group}

        val_group = TrainValDataGroup(
            name=VALIDATION, parent_dir=self.path, split=split.index
        )
        val_group.create(dataset=split.val_dataset)
        self.data_groups[VALIDATION] = {split.index: val_group}

    def create(self) -> None:
        """
        Creates the MAPS directory if it does not already exist.

        Raises:
            ClinicaDLConfigurationError: If the directory already exists.
        """
        if self.exists():
            raise ClinicaDLConfigurationError(
                f"Maps directory ({self.path})already exists."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        self.groups_dir.mkdir(parents=True)
        self._write_requirements_version()

    def _write_requirements_version(self) -> None:
        """Writes the installed Python packages (via `pip freeze`) to `environment.txt`."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            raise ClinicaDLConfigurationError(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def read_maps(self) -> dict:
        """Reads the maps.json file."""
        if not self.maps_json.is_file():
            raise ClinicaDLConfigurationError("Could not find maps.json")

        with open(self.maps_json, "r") as file:
            x = json.load(file)
            return json.loads(x)

    def caps_dir(self) -> Path:  # TODO: to change !
        return self.read_maps()["caps_dir"]
