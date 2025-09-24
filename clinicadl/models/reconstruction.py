from __future__ import annotations

import torch

from clinicadl.data.dataloader import Batch
from clinicadl.losses import LossOrConfig
from clinicadl.networks import NetworkOrConfig
from clinicadl.optim.optimizers import OptimizerOrConfig
from clinicadl.utils.typing import PathType

from .base import ClinicaDLModel
from .supervised import SupervisedModel, SupervisedModelConfig


class ReconstructionModelConfig(SupervisedModelConfig):
    """
    Config class for ReconstructionModel.
    """

    @classmethod
    def _get_class(cls) -> type[ClinicaDLModel]:
        """Returns the class associated to this config class."""
        return ReconstructionModel


class ReconstructionModel(SupervisedModel):
    """
    A vanilla reconstruction model, to work with simple AutoEncoders like
    :py:class:`~clinicadl.networks.nn.AutoEncoder`.

    Only the :py:meth:`forward_step` differs from :py:class:`~clinicadl.model.SupervisedModel`.

    Parameters
    ----------
    network : NetworkOrConfig
        The autoencoder, passed as a :py:class:`torch.nn.Module` or
        a :py:mod:`config class <clinicadl.networks.config>`.
    loss : LossOrConfig
        The reconstruction loss function, passed as a ``callable``, that returns a **1-item** :py:class:`~torch.Tensor`,
        or a :py:mod:`config class <clinicadl.losses.config>`.
    optimizer : OptimizerOrConfig
        The optimizer, passed as a :py:class:`torch.optim.Optimizer` or
        a :py:mod:`config class <clinicadl.optim.optimizers.config>`.

    See Also
    --------
    :py:class:`~clinicadl.model.SupervisedModel`
        For supervised training.
    """

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerOrConfig,
    ):
        self._config = ReconstructionModelConfig(
            network=network, loss=loss, optimizer=optimizer
        )
        objects = self._config.get_objects()
        self.network = objects["network"]
        self.loss = objects["loss"]
        self.optimizer = objects["optimizer"]

    def forward_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs a pass forward in the autoencoder and a comparison with the input image.

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
        outputs = self.network(images)

        loss = self.loss(outputs, images)

        return loss

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs) -> ReconstructionModel:
        """
        Creates a model from a ``JSON`` file saved with
        :py:meth:`write_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        kwargs : Any
            To pass directly any argument that ``ReconstructionModel``
            will not be able to read in the ``JSON`` file. Useful when you don't
            use config classes.

        Returns
        -------
        ReconstructionModel
            The model instantiated from the input file.
        """
        config: ReconstructionModelConfig = ReconstructionModelConfig.from_json(
            json_path, **kwargs
        )

        return cls(network=config.network, loss=config.loss, optimizer=config.optimizer)
