from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.utils import tsv_to_df
from clinicadl.dictionary.words import GROUPS, PARTICIPANT_ID
from clinicadl.experiment_manager import ExperimentManager
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


class Predictor:
    def __init__(self, reader: MapsReader, model: ClinicaDLModel):
        """TO COMPLETE"""
        self.reader = reader
        self.model = model

    def predict(self, dataset_test: CapsDataset, split: Split):
        """TO COMPLETE"""
        pass

    def _check_leakage(self, dataset_test: CapsDataset):
        """Checks that no intersection exist between the participants used for training and those used for testing."""

        df_train_val = self.reader.get_train_val_df()
        df_test = dataset_test.df

        participants_train = set(df_train_val[PARTICIPANT_ID].values)
        participants_test = set(df_test[PARTICIPANT_ID].values)
        intersection = participants_test & participants_train

        if len(intersection) > 0:
            raise ClinicaDLDataLeakageError(
                "Your evaluation set contains participants who were already seen during "
                "the training step. The list of common participants is the following: "
                f"{intersection}."
            )

    def test(self):
        """Computes the predictions and evaluation metrics."""
        pass

    def _test_loader(self):
        """Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files."""
        pass

    def _compute_latent_tensor(self):
        """Compute the output tensors and saves them in the MAPS."""
        pass

    @torch.no_grad()
    def _compute_output_nifti(self):
        """omputes the output nifti images and saves them in the MAPS."""
        pass

    @torch.no_grad()
    def _compute_output_tensors(self):
        """Compute the output tensors and saves them in the MAPS."""
        pass

    def _ensemble_prediction(self):
        """Computes the results on the image-level."""
        pass
