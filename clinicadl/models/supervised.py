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
from clinicadl.utils.config import (
    ObjectConfig,
    ObjectOrConfig,
)
from clinicadl.utils.dictionary.words import IMAGE, LABEL, OPTIMIZER
from clinicadl.utils.objects import HasConfig

from .base import ClinicaDLModel

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch


class SupervisedModelConfig(ObjectConfig["SupervisedModel"]):
    """
    Config class for SupervisedModel.

    This class checks the network, the loss and the optimizer,
    converts them if they are passed via config classes, and also
    takes care of saving in .json format.
    """

    network: ObjectOrConfig[nn.Module, NetworkConfig] = Field(
        reader=ObjectOrConfig.build_reader(get_network_from_dict)
    )
    loss: ObjectOrConfig[Loss, LossConfig] = Field(
        reader=ObjectOrConfig.build_reader(get_loss_function_from_dict)
    )
    optimizer: ObjectOrConfig[optim.Optimizer, OptimizerConfig] = Field(
        reader=ObjectOrConfig.build_reader(get_optimizer_from_dict)
    )
    inferer: Inferer = Field(reader=get_inferer_from_dict)

    @field_validator("network", "loss", "optimizer", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        """
        Converts a value to a ObjectOrConfig.
        """
        return ObjectOrConfig.from_value(v)

    @classmethod
    def _get_class(cls) -> type[ClinicaDLModel]:
        """Returns the class associated to this config class."""
        return SupervisedModel


class SupervisedModel(HasConfig[SupervisedModelConfig], ClinicaDLModel):
    """
    A vanilla supervised model, for usual **classification**, **regression**,
    or **segmentation** task.

    Parameters
    ----------
    network : NetworkOrConfig
        The neural network, passed as a :py:class:`torch.nn.Module` or
        a :py:mod:`config class <clinicadl.networks.config>`.
    loss : LossOrConfig
        The loss function, passed as a ``callable``, that returns a **1-item** :py:class:`~torch.Tensor`,
        or a :py:mod:`config class <clinicadl.losses.config>`.

        .. important::
            The loss function must have a :torch:`PyTorch style <nn.html#loss-functions>`,
            with an attribute named ``reduction`` that can be set to ``none``.

    optimizer : OptimizerConfig
        The optimizer, passed as a :py:mod:`config class <clinicadl.optim.optimizers.config>`.

    See Also
    --------
    :py:class:`~clinicadl.models.ReconstructionModel`
        For image reconstruction.
    """

    _config_type = SupervisedModelConfig

    network: nn.Module
    loss: Loss
    optimizer: optim.Optimizer

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerConfig,
        inferer: Inferer = SimpleInferer(),
    ):
        super().__init__()
        self.config = self._config_type(
            network=network, loss=loss, optimizer=optimizer, inferer=inferer
        )
        self.network = self.config.network.get_object()
        self.loss = self.config.loss.get_object()
        self.optimizer = self.config.optimizer.get_object(network=self.network)
        self.inferer = self.config.inferer

    def forward_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs a classical supervised forward step and returns the computed loss.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.

        Returns
        -------
        torch.Tensor
            The computed loss, as a **1-item** :py:class:`torch.Tensor`.
        """
        images = batch.get_field(IMAGE, dtype=torch.float32)
        labels = batch.get_field(LABEL, ensure_channel_dim=True, dtype=torch.float32)

        outputs = self.network(images)

        loss = self.loss(outputs, labels)

        return loss

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
            The optimizers, as defined in :py:meth:`build_optimizers`.
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
        Returns the loss function, that will be computed
        on the validation set.

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
        input_data: torch.Tensor,
    ) -> str:
        """
        Returns a summary of the neural network, produced by
        `torchinfo <https://github.com/TylerYep/torchinfo>`_.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data to pass to the neural network to build the summary.

        Returns
        -------
        str
            The summary.
        """
        from torchinfo import summary

        summary_ = summary(
            self.network,
            input_data=input_data,
        )

        return str(summary_)
