from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.utils import tsv_to_df
from clinicadl.dictionary.words import GROUPS, PARTICIPANT_ID
from clinicadl.experiment_manager.data_group import DataGroup
from clinicadl.model import ClinicaDLModel
from clinicadl.splitter.split import Split
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)


class MapsReader:
    maps_path: Path

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

    def _load_data_group(self, name: str, split: Optional[int] = None) -> DataGroup:
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
        return ClinicaDLModel()

    def _write_network_weights(self):
        """TO COMPLETE"""
        pass

    def _write_optim_weights(self):
        """TO COMPLETE"""
        pass

    def write_tensor(self):
        """TO COMPLETE"""
        pass

    def optimizer_path(self, split: int, resume: bool = False) -> Path:
        """TO COMPLETE"""

        checkpoint_path = (
            self.maps_path / f"split-{split}" / "tmp" / "optimizer.pth.tar"
        )
        return checkpoint_path

    def checkpoint_path(self, split: int, resume: bool = False):
        checkpoint_path = (
            self.maps_path / f"split-{split}" / "tmp" / "checkpoint.pth.tar"
        )
        return checkpoint_path
