from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.amp as amp
import torch.nn as nn
import torch.optim as optim

from clinicadl.utils.objects import JsonReaderWriter

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch, BatchType
    from clinicadl.losses.types import Loss, LossType


class Model(JsonReaderWriter, ABC, nn.Module):
    """
    The base model from which every model that works with ``ClinicaDL`` must inherit.

    ``Model`` inherits itself from :py:class:`torch.nn.Module`. So you can classically define
    your neural networks in the ``__init__`` method (don't forget to call ``super().__init__()`` first!).

    Besides, the following methods must be overwritten:

    - :py:meth:`forward_step`: defines the forward logic during training;
    - :py:meth:`backward_step`: defines the gradients computation logic;
    - :py:meth:`optimization_step`: defines the optimization logic;
    - :py:meth:`evaluation_step`: defines the evaluation logic;
    - :py:meth:`prediction_step`: defines the prediction logic;
    - :py:meth:`build_optimizers`: to build the optimizers used for training;
    - :py:meth:`get_loss_functions`: to access the loss functions used during training.

    You can also overwrite :py:meth:`get_summary` to give a description of your neural network(s).

    .. tip::
        Since rewriting all these methods can be tedious, feel free to inherit from an existing ``Model`` with shared logic,
        and rewrite only the relevant methods.

    See Also
    --------
    :py:class:`~clinicadl.models.SupervisedModel`
        A ``Model`` for supervised training.
    :py:class:`~clinicadl.models.ReconstructionModel`
        A ``Model`` for image reconstruction.
    """

    @abstractmethod
    def forward_step(self, batch: BatchType) -> LossType:
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
        loss: LossType,
        grad_scaler: amp.GradScaler = amp.GradScaler(enabled=False),
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
        optimizers: dict[str, optim.Optimizer],
        grad_scaler: amp.GradScaler = amp.GradScaler(enabled=False),
    ) -> None:
        """
        Performs the optimization step using the gradients accumulated in
        :py:meth:`backward_step`.

        .. note::
            ``ClinicaDL`` takes care of zeroing gradients after this step, using
            the optimizers returned by :py:meth:`get_optimizers`.

        Parameters
        ----------
        optimizers : dict[str, torch.optim.Optimizer]
            The optimizers, as defined in :py:meth:`build_optimizers`.
        grad_scaler : GradScaler, default=GradScaler(enabled=False)
            A potential :torch:`torch.amp.GradScaler <amp.html#gradient-scaling>` used to scale gradients.
        """

    @abstractmethod
    def evaluation_step(self, batch: BatchType) -> Batch:
        """
        Performs the evaluation step where a validation/test batch is passed through
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
    def prediction_step(self, batch: BatchType) -> Batch:
        """
        Performs inference on a batch.

        As opposed to :py:meth:`evaluation_step`, no metrics will be computed on the outputs. This method is to
        use the model for inference once it has been trained and tested.

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
    def build_optimizers(self) -> dict[str, optim.Optimizer]:
        """
        To build optimizers that will be used during training.

        All optimizers must be given a name.

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

    def get_summary(
        self,
        input_data: BatchType,
    ) -> str:
        """
        Returns a summary of your neural network, produced by
        `torchinfo <https://github.com/TylerYep/torchinfo>`_ for example.

        If this method is not implemented, the ``nn_summary.txt`` will not be
        created.

        Parameters
        ----------
        input_data : BatchType
            Input data to pass to the neural network to build the summary.

        Returns
        -------
        str
            The summary.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Resets the neural network(s) weights.
        """
        _reset_recursively(self)


def _reset_recursively(module: nn.Module) -> None:
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        _reset_recursively(layer)
