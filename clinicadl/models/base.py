from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.optim as optim
from torch.amp import GradScaler

from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.losses.types import Loss
from clinicadl.utils.device import DeviceType
from clinicadl.utils.objects import JsonReaderWriter
from clinicadl.utils.typing import PathType


class ClinicaDLModel(JsonReaderWriter, ABC):
    """
    The base model from which every model that works with ``ClinicaDL`` must inherit.

    The following methods must be overwritten:

    - :py:meth:`forward_step`: defines the forward logic during training;
    - :py:meth:`backward_step`: defines the gradients computation logic;
    - :py:meth:`optimization_step`: defines the optimization logic;
    - :py:meth:`evaluation_step`: defines the evaluation logic;
    - :py:meth:`get_optimizers`: to access the optimizers used for training;
    - :py:meth:`get_loss_functions`: to access the loss functions used during training;
    - :py:meth:`to`: to move the model on a specific device and/or cast the model to a specific datatype and/or memory format;
    - :py:meth:`train`: to set the model in training mode;
    - :py:meth:`eval`: to set the model in evaluation mode;
    - :py:meth:`state_dict`: to get a checkpoint of the model;
    - :py:meth:`load_state_dict`: to load a checkpoint of the model.

    You can also overwrite :py:meth:`get_architecture` and :py:meth:`get_summary`
    to give descriptions of your neural network(s). If these two methods are not implemented,
    the associated files in your :py:class:`MAPS directory <clinicadl.io.Maps>` will remain empty.

    .. tip::
        Since rewriting all these methods can be tedious, feel free to inherit from an existing ``ClinicaDLModel`` with shared logic,
        and rewrite only the relevant methods.

    See Also
    --------
    :py:class:`~clinicadl.models.SupervisedModel`
        A ``ClinicaDLModel`` for supervised training.
    :py:class:`~clinicadl.models.ReconstructionModel`
        A ``ClinicaDLModel`` for image reconstruction.
    """

    @abstractmethod
    def forward_step(
        self, batch: BatchType
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Performs the training forward step using the provided batch of data and returns
        the computed loss.

        Several losses can be computed during this step.

        It is on this loss(es) that the gradients will be computed.

        .. note::
            No need to send tensors to another device, or to implement Automatic Mixed Precision,
            ``ClinicaDL`` takes care of this.

        Parameters
        ----------
        batch : BatchType
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of ``Batch``
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        Union[torch.Tensor, dict[str, torch.Tensor]]
            The computed loss(es), as a **1-item** :py:class:`torch.Tensor`, or a dictionary of such ``Tensors``.
        """

    @abstractmethod
    def backward_step(
        self,
        loss: Union[torch.Tensor, dict[str, torch.Tensor]],
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        """
        Performs gradient computation using the loss(es) returned by :py:meth:`forward_step`.

        Parameters
        ----------
        loss : Union[torch.Tensor, dict[str, torch.Tensor]]
            The loss(es) on which gradient will be computed.
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """

    @abstractmethod
    def optimization_step(
        self,
        grad_scaler: GradScaler = GradScaler(enabled=False),
    ) -> None:
        """
        Performs the optimization step using the gradients accumulated in
        :py:meth:`backward_step`.

        .. note::
            ``ClinicaDL`` takes care of zeroing gradients after this step, using
            the optimizers returned by :py:meth:`get_optimizers`.

        Parameters
        ----------
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """

    @abstractmethod
    def evaluation_step(self, batch: BatchType) -> Batch:
        """
        Performs the evaluation step where a validation batch is passed through
        the neural network and an output batch is inferred.

        The output batch contains :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`
        on which the :py:mod:`metrics <clinicadl.metrics>` will be computed.

        .. note::
            No need to send tensors to another device or to wrap your evaluation logic in the ``torch.no_grad()`` context manager,
            ``ClinicaDL`` takes care of this.

        Parameters
        ----------
        batch : BatchType
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`. It can either a
            :py:class:`~clinicadl.data.dataloader.Batch`, or a ``tuple`` of ``Batch``
            (e.g. if you use :py:class:`~clinicadl.data.datasets.PairedDataset`).

        Returns
        -------
        Batch
            The output :py:class:`~clinicadl.data.dataloader.Batch`.

            .. important::
                Even if the input batch is a ``tuple`` of :py:class:`~clinicadl.data.dataloader.Batch`,
                the output must be a single :py:class:`~clinicadl.data.dataloader.Batch`. Metrics will be
                computed on each element of this output batch.
        """

    @abstractmethod
    def get_optimizers(self) -> dict[str, optim.Optimizer]:
        """
        To retrieve all optimizers used during training.

        All optimizers must be given a name.

        This methods enables ``ClinicaDL`` to perform operations
        on your optimizers, such as :torch:`learning rate scheduling <optim.html#how-to-adjust-learning-rate>`.

        Returns
        -------
        dict[str, optim.Optimizer]
            The optimizers and their names.
        """

    @abstractmethod
    def get_loss_functions(self) -> dict[str, Loss]:
        """
        To retrieve loss functions used during training.

        All loss functions must be given a name.

        This method enables ``ClinicaDL`` to compute losses on the validation set.

        .. important::
            All loss functions must have a :torch:`PyTorch style <nn.html#loss-functions>`, i.e. a
            callable that returns a :py:class:`torch.Tensor` and with an attribute named ``reduction``
            that can be set to ``"none"`` in order to compute the validation loss at the image level
            (otherwise, the reduction is done at the batch level, so image-level results are not accessible).

        Returns
        -------
        dict[str, Loss]
            The loss functions and their names.
        """

    @abstractmethod
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

        See Also
        --------
        :py:meth:`torch.nn.Module.to`
        """

    @abstractmethod
    def train(self) -> None:
        """
        To set the model in training mode.

        See Also
        --------
        :py:meth:`torch.nn.Module.train`
        """

    @abstractmethod
    def eval(self) -> None:
        """
        To set the model in evaluation mode.

        See Also
        --------
        :py:meth:`torch.nn.Module.eval`
        """

    @abstractmethod
    def state_dict(
        self,
    ) -> dict[str, Any]:
        """
        To save a checkpoint of the weights of the neural network(s),
        as well as the state of the optimizer(s).

        Returns
        ----------
        dict[str, Any]
            A dictionary containing the states of the neural network(s) and
            the optimizer(s).
        """

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        """
        To load a checkpoint of the neural network(s) and the optimizer(s).

        This method must define the logic to read an output of :py:meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict[str, Any]
            The state returned by :py:meth:`state_dict`.
        """

    def get_architecture(self) -> str:
        """
        Returns the architecture of your neural network.

        If this method is not implemented, the ``architecture.log`` file of your
        :py:class:`MAPS directory <clinicadl.io.Maps>` will be empty.

        Returns
        -------
        str
            The string representation of the architecture.
        """
        raise NotImplementedError()

    def get_summary(
        self,
        input_data: torch.Tensor,
    ) -> str:
        """
        Returns a summary of your neural network, produced by
        `torchinfo <https://github.com/TylerYep/torchinfo>`_ for example.

        If this method is not implemented, the ``nn_summary.txt`` file of your
        :py:class:`MAPS directory <clinicadl.io.Maps>` will be empty.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data to pass to the neural network to build the summary.

        Returns
        -------
        str
            The summary.
        """
        raise NotImplementedError()
