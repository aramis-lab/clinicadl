from contextlib import redirect_stdout
from typing import Optional

from torchsummary import summary

from clinicadl.IO.maps.maps import Maps
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.utils.computational.config import ComputationalConfig

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

        self.n_batch = len(split.train_loader)
        self.split = split
        self.stop = False
        self.epoch = 0
        self.batch = 0

    def write_torchsummary(self):
        with open(
            self.maps.training.splits[self.split.index].torchsummary_txt, "w"
        ) as f:
            with redirect_stdout(f):
                summary(
                    self.model.network,
                    input_size=self.model._input_size,
                    batch_size=self.n_batch,
                    device=self.comp.device.type,
                )
