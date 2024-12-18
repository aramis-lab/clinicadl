from typing import Optional

import pandas as pd
import torch

from clinicadl.data.datasets import CapsDataset
from clinicadl.experiment_manager import ExperimentManager
from clinicadl.model import ClinicaDLModel
from clinicadl.splitter.split import Split
from clinicadl.utils.exceptions import ClinicaDLDataLeakageError


class MapsReader:
    def _check_data_group(self, data_group: str) -> bool:
        """Check if a data group is already available if other arguments are None."""
        return True

    def _create_data_group(self, data_group: str):
        """creates a new data_group."""
        pass

    def get_group_info(self, data_group: str, split: Optional[Split]):
        """Gets information from corresponding data group
        (list of participant_id / session_id + configuration parameters).
        split is only needed if data_group is train or validation."""
        pass

    def get_group_df(
        self, data_group: str, split: Optional[Split] = None
    ) -> pd.DataFrame:
        """Gets information from corresponding data group
        (list of participant_id / session_id).
        split is only needed if data_group is train or validation."""

        return pd.DataFrame()


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

        df_train = self.reader.get_group_df("train+validation")
        df_test = dataset_test.df

        participants_train = set(df_train.participant_id.values)
        participants_test = set(df_test.participant_id.values)
        intersection = participants_test & participants_train

        if len(intersection) > 0:
            raise ClinicaDLDataLeakageError(
                "Your evaluation set contains participants who were already seen during "
                "the training step. The list of common participants is the following: "
                f"{intersection}."
            )

    def _write_data_group(self):
        """Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).
        """
        pass

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
