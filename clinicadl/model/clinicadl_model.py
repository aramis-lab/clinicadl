import io
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer

from clinicadl.data.dataloader import Batch
from clinicadl.losses.config import LossConfig, get_loss_function_config
from clinicadl.losses.types import Loss
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.networks.config import NetworkConfig, get_network_config
from clinicadl.optim.optimizers.config import OptimizerConfig, get_optimizer_config
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType

# import idr_torch


class ClinicaDLModel:
    def __init__(
        self,
        network: Union[nn.Module, NetworkConfig],
        loss: Union[Loss, LossConfig],
        optimizer: Union[Optimizer, OptimizerConfig],
    ):
        if isinstance(network, NetworkConfig):
            self.network = network.get_object()
            self._network_config = network
        else:
            self.network = network

        if isinstance(loss, LossConfig):
            self.loss = loss.get_object()
            self._loss_config = loss
        else:
            self.loss = loss

        if isinstance(optimizer, OptimizerConfig):
            self.optimizer = optimizer.get_object(self.network)
            self._optimizer_config = optimizer
        else:
            self.optimizer = optimizer

        self.memory_format = torch.channels_last
        self.non_blocking: bool = False
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._input_size = None
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
    def from_json(cls, json_path: PathType):
        """
        Reads a JSON file and returns a ClinicaDLModel instance.
        """
        json_path = Path(json_path)
        dict_ = read_json(json_path=json_path)

        return cls.from_dict(dict_)

    @classmethod
    def from_dict(cls, dict_: dict):
        network_config = get_network_config(**dict_["network"])
        loss_config = get_loss_function_config(**dict_["loss"])
        optimizer_config = get_optimizer_config(**dict_["optimizer"])
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
        loss = loss_config.get_object()
        network = network_config.get_object()
        optimizer = optimizer_config.get_object(network=network)

        model = ClinicaDLModel(network, loss, optimizer)

        model._network_config = network_config
        model._loss_config = loss_config
        model._optimizer_config = optimizer_config

        return model

    def load_optim_state_dict(self, optimizer_path: Path):
        checkpoint_state = torch.load(
            optimizer_path, map_location=self.device, weights_only=True
        )
        self.optimizer.load_state_dict(checkpoint_state["optimizer"])
        # self.network.load_optim_state_dict(
        #     self.optimizer, checkpoint_state["optimizer"]
        # )

    def load_network_state_dict(self, model_path: Path):
        model_state = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        self.network.load_state_dict(model_state["model"])

        return model_state["epoch"]

    def training_step(self, data: Batch, device: torch.device) -> torch.Tensor:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """
        labels = data.get_labels().to(device).float()
        images = data.get_images().to(device)

        self._input_size = images.shape[1:]

        outputs = self.network(images)
        labels = labels.unsqueeze(dim=-1)

        loss = self.loss(outputs, labels)

        return loss

    def validation_step(
        self, data: Batch, device: torch.device, metrics: MetricsHandler
    ) -> MetricsHandler:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """
        labels = data.get_labels().to(device).float()
        images = data.get_images().to(device)

        outputs = self.network(images)
        labels = labels.unsqueeze(dim=-1)
        metrics(outputs, labels)

        return metrics

    def train(self):
        self.network.to(self.device)
        self.network.to(
            non_blocking=self.non_blocking
        )  # memory_format=self.memory_format (for ddp)
        self.network.train()

    def write_json(self, json_path: PathType, overwrite: bool = False) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        json_path = Path(json_path)

        if (
            not self._network_config
            or not self._loss_config
            or not self._optimizer_config
        ):
            raise ValueError(
                "Network, loss, and optimizer configs must be set before writing to JSON."
            )

        net_json = {"network": self._network_config.to_dict()}
        loss_json = {"loss": self._loss_config.to_dict()}
        optimizer_json = {"optimizer": self._optimizer_config.to_dict()}
        write_json(
            json_path=json_path,
            data={**net_json, **loss_json, **optimizer_json},
            overwrite=overwrite,
        )

    def write_architecture_log(self, log_path: PathType) -> None:
        with open(log_path, "w") as f:
            print(self.network, file=f)
