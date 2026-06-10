from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, field_validator

from clinicadl.infer import Inferer, SimpleInferer
from clinicadl.infer.factory import get_inferer_from_dict
from clinicadl.losses.config import LossConfig
from clinicadl.losses.factory import get_loss_function_from_dict
from clinicadl.losses.types import Loss, LossOrConfig
from clinicadl.networks.config import NetworkConfig
from clinicadl.networks.factory import get_network_from_dict
from clinicadl.networks.types import NetworkOrConfig
from clinicadl.optim.optimizers.config import OptimizerConfig
from clinicadl.optim.optimizers.factory import get_optimizer_from_dict
from clinicadl.utils.config import ObjectOrConfig
from clinicadl.utils.dictionary.words import IMAGE, OPTIMIZER

from .base import Model

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch
    from clinicadl.data.structures import Sample


class VanillaModelConfig:
    """
    Config class for vanilla supervised and reconstruction models.

    This class checks the network, the loss and the optimizer,
    converts them if they are passed via config classes, and also
    takes care of saving in .json format.
    """

    network: ObjectOrConfig[nn.Module, NetworkConfig] = Field(
        json_schema_extra={"reader": ObjectOrConfig.build_reader(get_network_from_dict)}
    )
    loss: ObjectOrConfig[Loss, LossConfig] = Field(
        json_schema_extra={
            "reader": ObjectOrConfig.build_reader(get_loss_function_from_dict)
        }
    )
    optimizer: ObjectOrConfig[optim.Optimizer, OptimizerConfig] = Field(
        json_schema_extra={
            "reader": ObjectOrConfig.build_reader(get_optimizer_from_dict)
        }
    )
    inferer: Inferer = Field(json_schema_extra={"reader": get_inferer_from_dict})

    @field_validator("network", "loss", "optimizer", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        """
        Converts a value to a ObjectOrConfig.
        """
        return ObjectOrConfig.from_value(v)


class VanillaModel(Model):
    """
    A base for vanilla supervised and reconstruction models.
    """

    _config_type: VanillaModelConfig

    network: nn.Module
    loss: Loss
    optimizer: optim.Optimizer

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerConfig,
        inferer: Inferer = SimpleInferer(),
        **kwargs,
    ):
        super().__init__()
        self.config: VanillaModelConfig = self._config_type(
            network=network,
            loss=loss,
            optimizer=optimizer,
            inferer=inferer,
            **kwargs,
        )
        self.network = self.config.network.get_object()
        self.loss = self.config.loss.get_object()
        self.optimizer = self.config.optimizer.get_object(network=self.network)
        self.inferer = self.config.inferer

    def backward_step(
        self,
        loss: torch.Tensor,
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        """
        Performs a classical gradient computation using the loss returned by :py:meth:`forward_step`.

        Parameters
        ----------
        loss : torch.Tensor
            The loss on which gradients will be computed.
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """
        grad_scaler.scale(loss).backward()

    def optimization_step(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        """
        Performs a classical optimization step using the gradients accumulated in
        :py:meth:`backward_step`.

        Parameters
        ----------
        optimizers : dict[str, torch.optim.Optimizer]
            The optimizer, as defined in :py:meth:`build_optimizers`.
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """
        grad_scaler.step(optimizers[OPTIMIZER])

    def evaluation_step(self, batch: Batch) -> Batch:
        """
        Passes the input images in the network and saves the output
        in the batch.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.

        Returns
        -------
        Batch
            The output :py:class:`~clinicadl.data.dataloader.Batch`.
        """
        return self.inferer(batch, self.network, input_dtype=torch.float32)

    def prediction_step(self, batch: Batch) -> Batch:
        """
        Performs a simple pass forward and saves the output. Exactly similar to :py:meth:`evaluation_step`.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.

        Returns
        -------
        Batch
            The output :py:class:`~clinicadl.data.dataloader.Batch`.
        """
        return self.evaluation_step(batch)

    def get_loss_functions(self) -> dict[str, Loss]:
        """
        Returns the loss function.

        Returns
        -------
        dict[str, Loss]
            The loss function, named ``"loss"``.
        """
        return {"loss": self.loss}

    def build_optimizers(self) -> dict[str, optim.Optimizer]:
        """
        Returns a new instance of the optimizer.

        Returns
        -------
        dict[str, optim.Optimizer]
            The optimizer, named ``"optimizer"``.
        """
        return {"optimizer": self.config.optimizer.get_object(network=self)}

    def get_summary(
        self,
        input_data: Batch,
    ) -> str:
        """
        Returns a summary of the neural network, produced by
        `torchinfo <https://github.com/TylerYep/torchinfo>`_.

        Parameters
        ----------
        input_data : Batch
            Input data to pass to the neural network to build the summary.

        Returns
        -------
        str
            The summary.
        """
        from torchinfo import summary

        summary_ = summary(
            self.network,
            input_data=input_data.get_field(IMAGE, dtype=torch.float32),
            batch_dim=0,
            verbose=0,
        )

        return str(summary_)
