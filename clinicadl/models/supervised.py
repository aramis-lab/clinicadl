from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from clinicadl.data.dataloader import Batch
from clinicadl.losses import Loss, LossOrConfig
from clinicadl.losses.config import get_loss_function_config
from clinicadl.networks import NetworkOrConfig
from clinicadl.networks.config import get_network_config
from clinicadl.optim.optimizers import OptimizerOrConfig
from clinicadl.optim.optimizers.config import OptimizerConfig, get_optimizer_config
from clinicadl.utils.config import (
    ConfigsOrObjects,
    FieldReadersType,
    ObjectConfig,
)
from clinicadl.utils.device import DeviceType
from clinicadl.utils.typing import PathType

from .base import ClinicaDLModel


class SupervisedModelConfig(ConfigsOrObjects):
    """
    Config class for SupervisedModel.

    This class checks the network, the loss and the optimizer,
    converts them if they are passed via config classes, and also
    takes care of saving in JSON format.
    """

    network: NetworkOrConfig
    loss: LossOrConfig
    optimizer: OptimizerOrConfig
    _FIELD_READERS: FieldReadersType = {
        "network": get_network_config,
        "loss": get_loss_function_config,
        "optimizer": get_optimizer_config,
    }

    def get_objects(self) -> dict[str, Any]:
        """
        Gets field values, a convert them to the underlying objects
        if they are config classes.
        """
        dict_ = {}
        for field, value in self:
            if isinstance(value, OptimizerConfig):
                dict_[field] = value.get_object(network=dict_["network"])
            elif isinstance(value, ObjectConfig):
                dict_[field] = value.get_object()
            else:
                dict_[field] = value

        return dict_

    @classmethod
    def _get_class(cls) -> type[ClinicaDLModel]:
        """Returns the class associated to this config class."""
        return SupervisedModel


class SupervisedModel(ClinicaDLModel):
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

    optimizer : OptimizerOrConfig
        The optimizer, passed as a :py:class:`torch.optim.Optimizer` or
        a :py:mod:`config class <clinicadl.optim.optimizers.config>`.

    See Also
    --------
    :py:class:`~clinicadl.models.ReconstructionModel`
        For image reconstruction.
    """

    network: nn.Module
    loss: Loss
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerOrConfig,
    ):
        self._config = SupervisedModelConfig(
            network=network, loss=loss, optimizer=optimizer
        )
        objects = self._config.get_objects()
        self.network = objects["network"]
        self.loss = objects["loss"]
        self.optimizer = objects["optimizer"]

    def forward_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs a classical supervised forward step and returns the computed loss.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of ``Batch``
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        torch.Tensor
            The computed loss, as a **1-item** :py:class:`torch.Tensor`.
        """
        images = batch.get_field("image", dtype=torch.float32)
        labels = batch.get_field("label", ensure_channel_dim=True, dtype=torch.float32)

        outputs = self.network(images)

        loss = self.loss(outputs, labels)

        return loss

    def optimization_step(
        self,
        loss: torch.Tensor,
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        """
        Performs a classical optimization step using the loss returned by
        :py:meth:`forward_step`.

        Parameters
        ----------
        loss : torch.Tensor
            The loss(es) on which gradient will be computed.
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """
        self.optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(self.optimizer)

    def evaluation_step(self, batch: Batch) -> Batch:
        """
        Passes the input images in the network and saves the output
        in the batch.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of ``Batch``
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        Batch
            The output :py:class:`~clinicadl.data.dataloader.Batch`.
        """
        images = batch.get_field("image", dtype=torch.float32)
        outputs = self.network(images)
        batch.add_field("output", outputs)

        return batch

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

    def get_optimizers(self) -> dict[str, optim.Optimizer]:
        """
        Returns the optimizer.

        Returns
        -------
        dict[str, optim.Optimizer]
            The optimizer, named ``"optimizer"``.
        """
        return {"optimizer": self.optimizer}

    def to(
        self,
        device: Optional[DeviceType] = None,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
        memory_format: Optional[torch.memory_format] = None,
    ) -> None:
        """
        To move the model on a specific device and/or cast
        the model to a specific datatype and/or memory format.

        Parameters
        ----------
        device : Optional[DeviceType], default=None
            The desired device. If ``None``, the model will stay on the current device.
        non_blocking : bool, default=False
            "When ``non_blocking`` is set to ``True``, the function attempts to perform the
            conversion asynchronously with respect to the host, if possible.
            This asynchronous behavior applies to both pinned and pageable memory."
            (see :torch:`PyTorch documentation <generated/torch.Tensor.to.html>`)
        dtype : Optional[torch.dtype], default=None
            The desired data type. If ``None``, the model will stay with the current dtype.
        memory_format : Optional[torch.memory_format], default=None
            The desired memory format. If ``None``, the model will stay with the current memory format.
        """
        self.network.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def train(self) -> None:
        """
        Set the neural network in training mode.
        """
        self.network.train()

    def eval(self) -> None:
        """
        Set the neural network in evaluation mode.
        """
        self.network.eval()

    def save_checkpoint(
        self,
        checkpoint_path: PathType,
        only_network_weights: bool = False,
    ) -> None:
        """
        To save a checkpoint of the weights of the neural network,
        and optionally a checkpoint of the state of the optimizer.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        only_network_weights : bool, default=False
            Whether to save only the weights of the neural network.
        """
        state_dict = {"network_state_dict": self.network.state_dict()}
        if not only_network_weights:
            state_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(state_dict, f=checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: PathType,
        device: DeviceType = torch.device("cpu"),
        only_network_weights: bool = False,
    ) -> None:
        """
        To load a checkpoint of the weights of the neural network,
        and optionally of the state of the optimizer.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        device : DeviceType, default=torch.device("cpu")
            On which device to load the checkpoint.
        only_network_weights : bool, default=False
            Whether to load only the weights of the neural network.
        """
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=device,
        )

        self.network.load_state_dict(checkpoint["network_state_dict"])
        if not only_network_weights:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def write_architecture_log(self, log_path: PathType) -> None:
        """
        Write the architecture of the model in a log file.

        Parameters
        ----------
        log_path : PathType
            The path to the log file.
        """
        with open(log_path, "w", encoding="utf-8") as f:
            print(self.network, file=f)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the model to a ``dict``.

        Returns
        -------
        dict[str, Any]
            The ``dict`` version of the model.
        """
        return self._config.to_dict()

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs) -> SupervisedModel:
        """
        Creates a model from a ``JSON`` file saved with
        :py:meth:`write_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        kwargs : Any
            To pass directly any argument that ``SupervisedModel``
            will not be able to read in the ``JSON`` file. Useful when you don't
            use config classes.

        Returns
        -------
        SupervisedModel
            The model instantiated from the input file.
        """
        config: SupervisedModelConfig = SupervisedModelConfig.from_json(
            json_path, **kwargs
        )

        return cls(network=config.network, loss=config.loss, optimizer=config.optimizer)
