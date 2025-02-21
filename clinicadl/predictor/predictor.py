from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.amp.autocast_mode import autocast

from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.words import GROUPS, PARTICIPANT_ID
from clinicadl.experiment_manager import ExperimentManager
from clinicadl.experiment_manager.maps_reader import MapsReader
from clinicadl.metrics.metrics import Metrics
from clinicadl.model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.splitter.split import Split
from clinicadl.tsvtools.utils import tsv_to_df
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)
from clinicadl.utils.typing import PathType


class Predictor:
    def __init__(
        self,
        maps_path: PathType,
        optim_config: OptimizationConfig = OptimizationConfig(),
        comp_config: ComputationalConfig = ComputationalConfig(),
    ):
        """TO COMPLETE"""

        self.reader = MapsReader(maps_path=maps_path)
        self.optim = optim_config
        self.comp = comp_config

    def test(
        self,
        dataloader: DataLoader,
        model: ClinicaDLModel,
        metrics: Metrics,
        epoch: int = 0,
    ):
        model.network.eval()
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                ##########
                images = (
                    torch.cat(list(sample.sample for sample in data), dim=0)
                    .unsqueeze(1)
                    .to(self.comp.device)
                )
                labels = (
                    torch.tensor([sample.label for sample in data], dtype=torch.float32)
                    .unsqueeze(1)
                    .to(self.comp.device)
                )  # TO REMOVE AND CHECK FOR MASK
                ##########
                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = model.network(images)
                    loss = model.loss(outputs, labels)

                # scaler.scale(loss_train).backward()
                metrics.val.compute(
                    batch=batch, epoch=epoch, data=(outputs, labels), loss=loss
                )
                print(loss)
                print(metrics.val.get_loss(batch=batch, epoch=epoch))
        model.network.train()
        return None

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
