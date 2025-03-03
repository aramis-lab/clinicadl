from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from clinicadl.losses import get_loss_function_from_config
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.networks import get_network_from_config
from clinicadl.networks.config import NetworkConfig
from clinicadl.optim import get_optimizer_from_config
from clinicadl.optim.optimizers import OptimizerConfig
from clinicadl.utils.computational.ddp import DDP


class ClinicaDLModel:
    def __init__(self, network: nn.Module, loss: Loss, optimizer: Optimizer):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self.network = DDP(
            self.network,
            fsdp=fully_sharded_data_parallel,
            amp=amp,
        )  # to check

    @classmethod
    def from_config(
        cls,
        network_config: NetworkConfig,
        loss_config: LossConfig,
        optimizer_config: OptimizerConfig,
    ):
        loss, _ = get_loss_function_from_config(loss_config)
        network, _ = get_network_from_config(network_config)
        optimizer, _ = get_optimizer_from_config(optimizer_config, network)

        return ClinicaDLModel(network, loss, optimizer)

    def load_optim_state_dict(self, optimizer_path: Path):
        checkpoint_state = torch.load(
            optimizer_path, map_location=self.network.device, weights_only=True
        )
        self.network.load_optim_state_dict(
            self.optimizer, checkpoint_state["optimizer"]
        )

    def load_state_dict(self, checkpoint_path: Path):
        checkpoint_state = torch.load(
            checkpoint_path, map_location=self.network.device, weights_only=True
        )
        self.network.load_state_dict(checkpoint_state["model"])

        return checkpoint_state["epoch"]

    def _init_from_maps(self, maps_path: Path):
        """TO COMPLETE"""
        # Load network and optimizer from maps_path, for tranqfer learning
        pass
