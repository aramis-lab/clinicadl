from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from clinicadl.losses import get_loss_function_config, get_loss_function_from_config
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.networks.config import NetworkConfig, get_network_from_config
from clinicadl.optim import get_optimizer_config, get_optimizer_from_config
from clinicadl.optim.optimizers import OptimizerConfig
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP
from clinicadl.utils.iotools.utils import update_json
from clinicadl.utils.typing import PathType

# import idr_torch


class ClinicaDLModel:
    def __init__(self, network: nn.Module, loss: Loss, optimizer: Optimizer):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self._network_config = None
        self._optimizer_config = None
        self._loss_config = None

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
    def from_dict(cls, dict_: dict):
        network_config = get_network_config(**dict_)
        loss_config = get_loss_function_config(**dict_)
        optimizer_config = get_optimizer_config(**dict_)
        return cls.from_config(
            network_config=network_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
        )

    @classmethod
    def from_config(
        cls,
        network_config: NetworkConfig,
        loss_config: LossConfig,
        optimizer_config: OptimizerConfig,
    ):
        loss, loss_config = get_loss_function_from_config(loss_config)
        network, network_config = get_network_from_config(network_config)
        optimizer, optimizer_config = get_optimizer_from_config(
            optimizer_config, network
        )

        model = ClinicaDLModel(network, loss, optimizer)

        model._network_config = network_config
        model._loss_config = loss_config
        model._optimizer_config = optimizer_config

        return model

    def load_optim_state_dict(self, optimizer_path: Path):
        checkpoint_state = torch.load(
            optimizer_path, map_location=self.device, weights_only=True
        )
        self.network.load_optim_state_dict(
            self.optimizer, checkpoint_state["optimizer"]
        )

    def load_state_dict(self, model_path: Path):
        model_state = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        self.network.load_state_dict(model_state["model"])

        return model_state["epoch"]

    def train(self):
        self.network.to(self.device)
        self.network.to(
            memory_format=self.memory_format, non_blocking=self.non_blocking
        )
        self.network.train()

    def write_info(self, json_path: PathType):
        json_path = Path(json_path)
        if self._network_config:
            update_json(json_path, self._network_config)

        if self._loss_config:
            update_json(json_path, self._loss_config)

        if self._optimizer_config:
            update_json(json_path, self._optimizer_config)
