import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV
from clinicadl.dictionary.words import BEST, GROUPS, PARTICIPANT_ID, SPLIT, TMP
from clinicadl.experiment_manager.data_group import DataGroup
from clinicadl.metrics.metrics import Metrics, TrainingMetrics
from clinicadl.model import ClinicaDLModel
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import tsv_to_df
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)
from clinicadl.utils.typing import PathType


class MapsReader:
    def __init__(self, maps_path: PathType) -> None:
        self.maps_path = Path(maps_path)

    def _create_maps(self, overwrite: bool = False):
        """TO COMPLETE"""
        if self.maps_path.is_dir() and len(list(self.maps_path.glob("*"))) != 0:
            if overwrite:
                shutil.rmtree(self.maps_path)
            else:
                raise ClinicaDLConfigurationError(
                    "You are trying to create a new maps folder but it already exists."
                )

        self.maps_path.mkdir(parents=True, exist_ok=True)
        self._write_requirements_version()
        self._write_maps_json()

    ###### GETTER ########

    def get_data_group(self, name: str, split: Optional[int] = None) -> DataGroup:
        """creates a new data_group."""
        data_group = DataGroup(name=name, split=split, maps_path=self.maps_path)
        if data_group.exists():
            return data_group

        raise ClinicaDLConfigurationError(
            f"Could not find data group {data_group.name}"
        )

    def get_train_val_df(self):
        """Loads the train and validation data groups."""
        path = self.maps_path / GROUPS / "train+validation.tsv"
        return tsv_to_df(path)

    def get_model(self) -> ClinicaDLModel:
        return ClinicaDLModel()  # type: ignore

    def get_metrics(self) -> TrainingMetrics:
        return TrainingMetrics()  # type: ignore

    ##### WRITERS #######
    def _write_network_weights(self):
        """TO COMPLETE"""
        pass

    def _write_optim_weights(self):
        """TO COMPLETE"""
        pass

    def write_tensor(self):
        """TO COMPLETE"""
        pass

    def _write_maps_json(self):
        """Writes the maps.json file."""
        if self.maps_json_path().is_file():
            raise ClinicaDLConfigurationError(
                "The maps.json file for this MPS already exists"
            )
        with (self.maps_json_path()).open(mode="w") as file:
            json.dump({}, file, indent=4)

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.maps_path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            raise ClinicaDLConfigurationError(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _create_data_group(
        self, name: str, caps_dataset: CapsDataset, split: Optional[int] = None
    ) -> DataGroup:
        """
        Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).
        """
        data_group = DataGroup(name=name, split=split, maps_path=self.maps_path)
        if data_group.exists():
            raise ClinicaDLConfigurationError(
                f"Data group {data_group.name} already exists, please give another name to your data group"
            )

        data_group.create(caps_dataset)
        return data_group

    def print_description_log(
        self,
        split: int,
        selection_metric: str,
        data_group: str,
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            data_group (str): name of the data group used for the task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
        """
        with self.description_log_path(split, selection_metric, data_group).open(
            mode="r"
        ) as f:
            content = f.read()

    ##### PATH #####

    def maps_json_path(self) -> Path:
        return self.maps_path / ("maps" + JSON)

    def train_val_tsv_path(self) -> Path:
        return self.maps_path / GROUPS / ("train+validation" + TSV)

    def information_log_path(self) -> Path:
        return self.maps_path / ("information" + LOG)

    def description_log_path(
        self, split: int, selection_metric: str, data_group: str
    ) -> Path:
        return self.data_group_path(split, selection_metric, data_group) / (
            "description" + LOG
        )

    def split_path(self, split: int) -> Path:
        return self.maps_path / (SPLIT + "-" + str(split))

    def optimizer_path(self, split: int, resume: bool = False) -> Path:
        """TO COMPLETE"""

        return self.split_path(split) / TMP / ("optimizer" + PTH + TAR)

    def checkpoint_path(self, split: int, resume: bool = False) -> Path:
        return self.split_path(split) / TMP / ("checkpoint" + PTH + TAR)

    def model_path(self, split: int, metric: str) -> Path:
        return self.split_path(split) / (BEST + "-" + metric) / ("model" + PTH + TAR)

    def best_metric_path(self, split: int, selection_metric: str) -> Path:
        return self.split_path(split) / f"best-{selection_metric}"

    def data_group_path(
        self, split: int, selection_metric: str, data_group: str
    ) -> Path:
        return self.best_metric_path(split, selection_metric) / data_group

    def prediction_tsv_path(
        self, split: int, selection_metric: str, data_group: str, mode: str
    ) -> Path:
        return (
            self.data_group_path(split, selection_metric, data_group)
            / f"{data_group}_{mode}_level_prediction.tsv"
        )
