from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from clinicadl.losses import get_loss_function_from_config
from clinicadl.losses.config import LossConfig
from clinicadl.losses.utils import Loss
from clinicadl.networks import get_network_from_config
from clinicadl.networks.config import NetworkConfig
from clinicadl.optim import get_optimizer_from_config
from clinicadl.optim.optimizers import OptimizerConfig
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP

# import idr_torch


class ClinicaDLModel:
    def __init__(self, network: nn.Module, loss: Loss, optimizer: Optimizer):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self.memory_format = torch.channels_last
        self.non_blocking: bool = False
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # if cluster.rank == 0: print(f'model: {network}')
        # if cluster.rank == 0: print('number of parameters: {}'.format(sum([p.numel()
        #                                       for p in network.parameters()])))

        # if cluster.rank == 0: print(f'Optimizer: {optimizer}')
        # self.network = DDP(
        #     self.network,
        #     fsdp=fully_sharded_data_parallel,
        #     amp=amp,
        # )  # to check

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

    def train(self):
        self.network.to(self.device)
        self.network.to(
            memory_format=self.memory_format, non_blocking=self.non_blocking
        )
        self.network.train()
