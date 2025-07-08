import io
import sys
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from clinicadl.data.dataloader import BatchType
from clinicadl.losses.config import LossConfig, get_loss_function_config
from clinicadl.losses.types import Loss
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.networks.config import NetworkConfig, get_network_config
from clinicadl.optim.optimizers.config import OptimizerConfig, get_optimizer_config
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.config import FieldReadersType, MultipleConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.json import read_json
from clinicadl.utils.typing import PathType


class ClinicaDLModelConfig(MultipleConfig):
    """
    Config class associated to ClinicaDLModel.
    """

    network: Optional[NetworkConfig] = None
    loss: Optional[LossConfig] = None
    optimizer: Optional[OptimizerConfig] = None
    _FIELD_READERS: FieldReadersType = {
        "network": get_network_config,
        "loss": get_loss_function_config,
        "optimizer": get_optimizer_config,
    }


class ClinicaDLModel:
    def __init__(
        self,
        network: Union[nn.Module, NetworkConfig],
        loss: Union[Loss, LossConfig],
        optimizer: Union[Optimizer, OptimizerConfig],
    ):
        self._config = ClinicaDLModelConfig()
        self._device = None

        if isinstance(network, NetworkConfig):
            self.network = network.get_object()
            self._config.network = network
        else:
            self.network = network

        if isinstance(loss, LossConfig):
            self.loss = loss.get_object()
            self._config.loss = loss
        else:
            self.loss = loss

        if isinstance(optimizer, OptimizerConfig):
            self.optimizer = optimizer.get_object(self.network)
            self._config.optimizer = optimizer
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
    def from_json(cls, json_path: PathType) -> None:
    @property
    def device(self) -> torch.device:
        """
        The device where is currently the neural network.
        """
        if not self._device:
            devices = set()
            for param in self.network.parameters():
                devices.add(param.device)
            for buffer in self.network.buffers():  # e.g. BatchNorm buffers
                devices.add(buffer.device)

            if len(devices) > 1:
                raise ClinicaDLConfigurationError(
                    "All the parameters of the neural network are not on the same device. "
                    f"Got for devices: {devices}"
                )
            self._device = devices.pop()

        return self._device

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
    def write_json(self, json_path: PathType, overwrite: bool = False) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        json_path = Path(json_path)

    def to(
        self,
        device: Optional[Union[str, torch.device, int]] = None,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
        channels_last: Optional[bool] = None,
    ) -> None:
        device_ = torch.device(device) if device else None

        self.network.to(
            device=device_,
            non_blocking=non_blocking,
            dtype=dtype,
            memory_format=channels_last,
        )

        if device:
            self._device = device_

        return model_state["epoch"]

    def training_step(self, data: BatchType, device: torch.device) -> torch.Tensor:
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
        self, data: BatchType, device: torch.device, metrics: MetricsHandler
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
    @classmethod
    def from_json(cls, json_path: PathType) -> ClinicaDLModel:
        """
        Creates a ``ClinicaDLModel`` instance from a ``JSON`` file.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        """
        config = ClinicaDLModelConfig.from_json(json_path)

        return cls(network=config.network, loss=config.loss, optimizer=config.optimizer)

    def write_json(self, json_path: PathType, overwrite: bool = False) -> None:
        """
        Writes the ``ClinicaDLModel`` parameters in a ``JSON`` file.

        .. warning::
            This method is relevant only if the ``ClinicaDLModel`` was
            instantiated with config classes. Otherwise, ``ClinicaDLModel``
            don't know what parameters to store.

        Parameters
        ----------
        json_path : PathType
            Path to the json file.
        overwrite : bool, default=True
            Whether to overwrite the json file if it exists.
        """
        self._config.write_json(json_path=json_path, overwrite=overwrite)

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        network_key: str = "model_state_dict",
        optimizer_key: str = "optimizer_state_dict",
    ) -> None:
        torch.save(
            {
                network_key: self.network.state_dict(),
                optimizer_key: self.optimizer.state_dict(),
            },
            f=checkpoint_path,
        )

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        network_key: str = "model_state_dict",
        optimizer_key: str = "optimizer_state_dict",
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, weights_only=True, map_location=self.device
        )
        self.network.load_state_dict(checkpoint[network_key])
        self.optimizer.load_state_dict(checkpoint[optimizer_key])

    def save_weights(
        self,
        checkpoint_path: Path,
    ) -> None:
        torch.save(self.network.state_dict(), f=checkpoint_path)

    def load_weights(
        self,
        checkpoint_path: Path,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.network.load_state_dict(checkpoint)
