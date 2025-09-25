from contextlib import redirect_stdout
from typing import Optional

from torchsummary import summary

from clinicadl.IO.maps.maps import Maps
from clinicadl.IO.maps.training.splits import EpochTmpDir
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.models import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.json import read_json, write_json

from ..utils.config.base import ClinicaDLConfig


class _TrainingState(ClinicaDLConfig):
    """
    Stores and manages the mutable state during a training session.

    This class acts as a centralized container for key objects and variables
    involved in training a ClinicaDL model on a specific data split. It keeps
    track of the current epoch, batch, whether training should stop, and holds
    references to core components such as data maps, metrics, model, optimizer,
    and computational configurations.

    Attributes
    ----------
    maps : Maps
        Provides access to dataset file paths and structure.
    metrics : MetricsHandler
        Handles computation and storage of performance metrics.
    model : ClinicaDLModel
        The neural network model being trained.
    optim : OptimizationConfig
        Configuration and state of the optimization process.
    comp : ComputationalConfig
        Computational settings such as device usage and resource limits.
    stop : bool
        Flag to indicate whether training should stop early.
    n_batch : int
        Number of batches in the current training epoch.
    split : Optional[Split]
        The current data split used for training.
    epoch : int
        The current epoch index.
    batch : int
        The current batch index within the epoch.

    Methods
    -------
    reset(split: Split)
        Initializes the training state for a new data split,
        resetting counters and setting batch count.
    """

    maps: Maps
    metrics: MetricsHandler
    model: ClinicaDLModel
    optim: OptimizationConfig
    comp: ComputationalConfig
    stop: bool = False
    n_batch: int = 0
    split: Optional[Split] = None
    epoch: int = 0
    batch: int = 0

    def reset(self, split: Split):
        """Reset the training state for a new training split."""

        if split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )

        self.n_batch = len(split.train_loader)
        self.split = split
        self.stop = False
        self.epoch = 0
        self.batch = 0

        # TODO:  temporary
        self.maps.load()
        if split.index not in self.maps.training.split_list:
            self.maps.training._create_split(split.index)

    def write_torchsummary(self):
        """Write the model summary to a text file in the maps directory."""
        if self.split is None:
            raise ValueError("The split has not been initialized.")

        with open(
            self.maps.training.splits[self.split.index].torchsummary_txt,
            "w",
            encoding="utf-8",
        ) as f:
            with redirect_stdout(f):
                summary(
                    self.model.network,
                    input_size=self.model._input_size,
                    batch_size=self.n_batch,
                    device=self.comp.device.type,
                )

    def save_checkpoint(self, checkpoint_path: EpochTmpDir) -> None:
        self.model.save_checkpoint(checkpoint_path.model)
        checkpoint_path.metrics._create()
        self.metrics.save(
            path=checkpoint_path.metrics.validation,
            details_path=checkpoint_path.metrics.validation_details,
        )
        write_json(checkpoint_path.stop, self.stop)

    def load_checkpoint(self, checkpoint_path: EpochTmpDir) -> None:
        self.maps.load()
        self.reset(self.split)
        self.model.load_checkpoint(checkpoint_path.model)
        self.metrics.load(
            path=checkpoint_path.metrics.validation,
            details_path=checkpoint_path.metrics.validation_details,
        )
        self.stop = read_json(checkpoint_path.stop)
        self.epoch = checkpoint_path.epoch + 1
