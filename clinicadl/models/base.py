from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Union

import torch
import torch.optim as optim
from torch.amp import GradScaler

from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.losses import Loss
from clinicadl.utils.device import DeviceType
from clinicadl.utils.exceptions import NotInterpretableJson
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType


class ImplementedModel(str, Enum):
    """Built-in ClinicaDLModels."""

    SUPERVISED = "SupervisedModel"
    RECONSTRUCTION = "ReconstructionModel"


class ClinicaDLModel(ABC):
    """
    The base model from which every model that works with ``ClinicaDL`` must inherit.

    The following methods must be overwritten:

    - :py:meth:`forward_step`: defines the forward logic during training;
    - :py:meth:`optimization_step`: defines the optimization logic;
    - :py:meth:`evaluation_step`: defines the evaluation logic;
    - :py:meth:`get_optimizers`: to access the optimizers used for training;
    - :py:meth:`get_loss_functions`: to access the loss functions used during training;
    - :py:meth:`to`: to move the model on a specific device and/or cast the model to a specific datatype and/or memory format;
    - :py:meth:`train`: to set the model in training mode;
    - :py:meth:`eval`: to set the model in evaluation mode;
    - :py:meth:`save_checkpoint`: to save a checkpoint of the model;
    - :py:meth:`load_checkpoint`: to load a checkpoint of the model;
    - :py:meth:`write_architecture_log`: to store a summary of the neural network architecture.

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
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
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
        Union[torch.Tensor, Sequence[torch.Tensor]]
            The computed loss(es), as a **1-item** :py:class:`torch.Tensor`, or a sequence of such ``Tensors``.
        """

    @abstractmethod
    def optimization_step(
        self,
        loss: Union[torch.Tensor, Sequence[torch.Tensor]],
        grad_scaler: GradScaler = GradScaler(enabled=False),
    ) -> None:
        """
        Performs the optimization step using the loss(es) returned by
        :py:meth:`forward_step`.

        Parameters
        ----------
        loss : Union[torch.Tensor, Sequence[torch.Tensor]]
            The loss(es) on which gradient will be computed.
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
    def save_checkpoint(
        self,
        checkpoint_path: PathType,
        only_network_weights: bool = False,
    ) -> None:
        """
        To save a checkpoint of the weights of the neural network,
        and optionally a checkpoint of the state of the optimizer.

        So here, the logic to save only the neural network weights,
        as well as the logic to save both the neural network and optimizer
        states, must be defined.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        only_network_weights : bool, default=False
            Whether to save only the weights of the neural network.
        """

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: PathType,
        device: DeviceType = torch.device("cpu"),
        only_network_weights: bool = False,
    ) -> None:
        """
        To load a checkpoint of the weights of the neural network,
        and optionally of the state of the optimizer.

        This method must define the logic to read the content saved
        with :py:meth:`save_checkpoint`.

        Parameters
        ----------
        checkpoint_path : PathType
            The path to the checkpoint.
        device : DeviceType, default=torch.device("cpu")
            On which device to load the checkpoint.
        only_network_weights : bool, default=False
            Whether to load only the weights of the neural network.
        """

    @abstractmethod
    def write_architecture_log(self, log_path: PathType) -> None:
        """
        To store a summary of the neural network architecture
        in a ``.log`` file.

        Parameters
        ----------
        log_path : PathType
            The path to the log file.
        """

    def write_json(self, json_path: PathType) -> None:
        """
        Writes the parameters of the model in a ``JSON`` file.

        Requires :py:meth:`to_dict` to be implemented.

        Parameters
        ----------
        json_path : PathType
            Path to the json file.
        """
        try:
            to_write = self.to_dict()
        except NotImplementedError:
            to_write = f"Custom model passed by the user: {type(self).__name__}"

        write_json(json_path, to_write)

    @staticmethod
    def from_json(json_path: PathType, **kwargs: Any) -> ClinicaDLModel:
        """
        Creates a model from a ``JSON`` file saved with
        :py:meth:`write_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        kwargs : Any
            To pass directly any argument that ``ClinicaDLModel``
            will not be able to read in the ``JSON`` file. Useful when you don't
            use config classes.

        Returns
        -------
        ClinicaDLModel
            The model instantiated from the input file.
        """
        dict_ = read_json(json_path)

        if not isinstance(dict_, dict):
            raise NotInterpretableJson(json_path, "ClinicaDLModel")

        try:
            name = dict_["name"]
        except KeyError as exc:
            raise KeyError(
                f"{str(json_path)} is not a valid json file for a ClinicaDLModel: it does not contain 'name'"
            ) from exc

        model = ImplementedModel(name).value

        # pylint: disable=import-outside-toplevel
        if model == ImplementedModel.SUPERVISED:
            from .supervised import SupervisedModel

            return SupervisedModel.from_json(json_path, **kwargs)
        elif model == ImplementedModel.RECONSTRUCTION:
            from .reconstruction import ReconstructionModel

            return ReconstructionModel.from_json(json_path, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the model to a ``dict``.

        This method must define the content of the dictionary that will
        enable ``ClinicaDLModel`` to recreate the model with
        the classmethod :py:meth:`from_dict`.

        The dictionary must contain at least a field 'name' with the
        type of ``ClinicaDLModel`` (e.g. "SupervisedModel").

        Returns
        -------
        dict[str, Any]
            The ``dict`` version of the model.
        """
        raise NotImplementedError(
            "Overwrite 'to_dict' to serialize you model and save it."
        )
