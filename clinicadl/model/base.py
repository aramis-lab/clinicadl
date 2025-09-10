from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn

from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.losses import Loss, LossOrConfig
from clinicadl.losses.config import get_loss_function_config
from clinicadl.networks import NetworkOrConfig
from clinicadl.networks.config import get_network_config
from clinicadl.optim.optimizers import OptimizerOrConfig
from clinicadl.optim.optimizers.config import get_optimizer_config
from clinicadl.utils.config import ConfigsOrObjects, FieldReadersType
from clinicadl.utils.device import DeviceType, check_device
from clinicadl.utils.typing import PathType


class ClinicaDLModel(ABC):
    """
    ``ClinicaDLModel`` defines the model, as well as its training and evaluation logic.

    A model is defined by a **neural network**, a **loss function**, and **an optimizer**.

    The user can overwrite this class to define its own ``ClinicaDLModel``. More precisely,
    the following methods must be overwritten:

    - :py:meth:`__init__`: where ``self.network`` (a :py:class:`torch.nn.Module), ``self.loss``
      (a callable that returns a py:class:`torch.Tensor`), and ``self.optimizer`` (a :py:class:`torch.optim.Optimizer`)
      must be defined;
    - :py:meth:`training_step`: that contains the training logic;
    - :py:meth:`evaluation_step`: that contains the evaluation logic;
    - :py:meth:`write_json`: to save the ``ClinicaDLModel`` in a ``JSON`` file;
    - :py:meth:`from_json`: to create a ``ClinicaDLModel`` from a ``JSON`` file.

    See Also
    --------
    :py:class:`clinicadl.model.SupervisedModel`
        A ``ClinicaDLModel`` for supervised training.
    :py:class:`clinicadl.model.ReconstructionModel`
        A ``ClinicaDLModel`` for image reconstruction.
    """

    network: nn.Module
    loss: Loss
    optimizer: torch.optim.Optimizer

    @abstractmethod
    def __init__(self):
        """
        ``network``, ``loss``, and ``optimizer`` must be defined here.
        """

    @abstractmethod
    def training_step(self, batch: BatchType) -> torch.Tensor:
        """
        Performs the training step on the model using the provided batch of data and returns
        the computed loss.

        It is on this loss that gradients will be computed.

        .. note::
            No need to send tensors to another device or to reset the gradients here,
            ``ClinicaDL`` takes care of this.

        Parameters
        ----------
        batch : BatchType
            The batch of :py:class:`~clinicadl.data.structures.DataPoint`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of :py:class:`~clinicadl.data.dataloader.Batch`
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        torch.Tensor
            The computed loss, as a **1-item** :py:class:`torch.Tensor`.
        """

    @abstractmethod
    def evaluation_step(self, batch: BatchType) -> Batch:
        """
        Performs the evaluation step where a validation batch is passed through
        the neural network and an output batch is inferred.

        The output batch contains :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`
        on which the :py:mod:`metrics <clinicadl.metrics` will be computed.

        .. note::
            No need to send tensors to another device or to wrap your evaluation logic in the ``torch.no_grad()`` context manager,
            ``ClinicaDL`` takes care of this.

        Parameters
        ----------
        batch : BatchType
            The batch of :py:class:`~clinicadl.data.structures.DataPoint`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of :py:class:`~clinicadl.data.dataloader.Batch`
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        torch.Tensor
            The computed loss, as a **1-item** :py:class:`torch.Tensor`.

            .. important::
                Even if the input batch is a ``tuple`` of :py:class:`~clinicadl.data.dataloader.Batch`,
                the output must be a single :py:class:`~clinicadl.data.dataloader.Batch`. Metrics will be
                computed on each element of this output batch.
        """

    @abstractmethod
    def write_json(self, json_path: PathType) -> None:
        """
        Writes the ``ClinicaDLModel`` parameters in a ``JSON`` file.

        The user must define here the content of the file that will
        enable ``ClinicaDLModel`` to recreate the same model with
        the class method :py:meth:`from_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the json file.
        """

    @classmethod
    @abstractmethod
    def from_json(cls, json_path: PathType) -> ClinicaDLModel:
        """
        Creates a ``ClinicaDLModel`` instance from a ``JSON`` file.

        This method must define the logic to read the content saved
        with :py:meth:`write_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.

        Returns
        -------
        ClinicaDLModel
            The model instantiated from the input file.
        """

    def save_checkpoint(
        self,
        checkpoint_path: PathType,
        network_key: str = "network_state_dict",
        optimizer_key: str = "optimizer_state_dict",
    ) -> None:
        """
        To save a checkpoint of the weights of the neural network
        and the state of the optimizer.

        This method will save a dictionary in ``checkpoint_path``,
        where the weights of the neural networks can be accessed via the key
        ``network_key``, and the optimizer state can be accessed via the key
        ``optimizer_key``.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        network_key : str, default="network_state_dict"
            The key to the neural network weights in the checkpoint dictionary.
        optimizer_key : str, default="optimizer_state_dict"
            The key to the optimizer state in the checkpoint dictionary.

        See Also
        --------
        :py:meth:`save_weights`
            To save only the weights of the neural network.
        """
        torch.save(
            {
                network_key: self.network.state_dict(),
                optimizer_key: self.optimizer.state_dict(),
            },
            f=checkpoint_path,
        )

    def load_checkpoint(
        self,
        checkpoint_path: PathType,
        device: Optional[DeviceType] = None,
        network_key: str = "network_state_dict",
        optimizer_key: str = "optimizer_state_dict",
    ) -> None:
        """
        To load a checkpoint of the weights of the neural network
        and the state of the optimizer.

        This method expects to find a dictionary in ``checkpoint_path``,
        where the weights of the neural networks can be accessed via the key
        ``network_key``, and the optimizer state can be accessed via the key
        ``optimizer_key``.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        device : Optional[DeviceType], default=None
            On which device to load the checkpoint. It must be the same device
            as the one where the neural network is.
        network_key : str, default="network_state_dict"
            The key to the neural network weights in the checkpoint dictionary.
        optimizer_key : str, default="optimizer_state_dict"
            The key to the optimizer state in the checkpoint dictionary.

        See Also
        --------
        :py:meth:`load_weights`
            To load only the weights of the neural network.
        """
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=check_device(device),
        )
        self.network.load_state_dict(checkpoint[network_key])
        self.optimizer.load_state_dict(checkpoint[optimizer_key])

    def save_weights(
        self,
        checkpoint_path: PathType,
    ) -> None:
        """
        To save a checkpoint of the weights of the neural network.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.

        See Also
        --------
        :py:meth:`save_checkpoint`
            To save the weights of the neural network, AND the state of the optimizer.
        """
        torch.save(self.network.state_dict(), f=checkpoint_path)

    def load_weights(
        self,
        checkpoint_path: PathType,
        device: Optional[DeviceType] = None,
    ) -> None:
        """
        To load a checkpoint of the weights of the neural network.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        device : Optional[DeviceType], default=None
            On which device to load the checkpoint. It must be the same device
            as the one where the neural network is.

        See Also
        --------
        :py:meth:`load_checkpoint`
            To load the weights of the neural network, AND the state of the optimizer.
        """
        checkpoint = torch.load(
            checkpoint_path, weights_only=True, map_location=check_device(device)
        )
        self.network.load_state_dict(checkpoint)

    def write_architecture_log(self, log_path: PathType) -> None:
        """
        To write the architecture of the model in a log file.

        Parameters
        ----------
        log_path : PathType
            The path to the log file.
        """
        with open(log_path, "w") as f:
            print(self.network, file=f)


class _BuiltinClinicaDLModelConfig(ConfigsOrObjects):
    """
    Config class associated to BuiltinClinicaDLModel.
    """

    network: NetworkOrConfig
    loss: LossOrConfig
    optimizer: OptimizerOrConfig
    _FIELD_READERS: FieldReadersType = {
        "network": get_network_config,
        "loss": get_loss_function_config,
        "optimizer": get_optimizer_config,
    }


class BuiltinClinicaDLModel(ClinicaDLModel):
    """
    Base class for built-in ``ClinicaDLModels``.
    """

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerOrConfig,
    ):
        self._config = _BuiltinClinicaDLModelConfig(
            network=network, loss=loss, optimizer=optimizer
        )
        self.network = self._config.get_object("network")
        self.loss = self._config.get_object("loss")
        self.optimizer = self._config.get_object("optimizer")

    def write_json(self, json_path: PathType) -> None:
        """
        Writes the ``ClinicaDLModel`` parameters in a ``JSON`` file.

        Parameters
        ----------
        json_path : PathType
            Path to the json file.
        """
        self._config.write_json(json_path=json_path)

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs: Any) -> ClinicaDLModel:
        """
        Creates a ``ClinicaDLModel`` instance from a ``JSON`` file.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        kwargs : Any
            To pass a custom ``network``, ``loss``, or ``optimizer`` if ``ClinicaDL```
            is not able to read the one in the ``JSON`` file.

        Returns
        -------
        ClinicaDLModel
            The model instantiated from the input file.
        """
        config = _BuiltinClinicaDLModelConfig.from_json(json_path, **kwargs)

        return cls(network=config.network, loss=config.loss, optimizer=config.optimizer)
