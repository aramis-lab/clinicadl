from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from clinicadl.data.dataloader import Batch
from clinicadl.losses.config import LossConfig, get_loss_function_config
from clinicadl.losses.types import Loss
from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.networks.config import NetworkConfig, get_network_config
from clinicadl.optim.optimizers.config import OptimizerConfig, get_optimizer_config
from clinicadl.utils.config import FieldReadersType, MultipleConfig
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

    @abstractmethod
    def training_step(
        self, data: Batch, device: Optional[Union[str, torch.device, int]] = None
    ) -> torch.Tensor:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """

    @abstractmethod
    def evaluation_step(
        self,
        data: Batch,
        metrics: ClinicaDLMetrics,
        device: Optional[Union[str, torch.device, int]] = None,
    ) -> ClinicaDLMetrics:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """

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
        device: Optional[Union[str, torch.device]] = None,
        network_key: str = "model_state_dict",
        optimizer_key: str = "optimizer_state_dict",
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=device,
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
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        self.network.load_state_dict(checkpoint)
