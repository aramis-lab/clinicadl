from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from clinicadl.utils.objects import HasConfig

from .base import ClinicaDLModel
from .supervised import SupervisedModel, SupervisedModelConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch


class ReconstructionModelConfig(SupervisedModelConfig):
    """
    Config class for ReconstructionModel.
    """

    @classmethod
    def _get_class(cls) -> type[ClinicaDLModel]:
        """Returns the class associated to this config class."""
        return ReconstructionModel


class ReconstructionModel(SupervisedModel, HasConfig[ReconstructionModelConfig]):
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

        .. important::
            The loss function must have a :torch:`PyTorch style <nn.html#loss-functions>`,
            with an attribute named ``reduction`` that can be set to ``none``.

    optimizer : OptimizerConfig
        The optimizer, passed as a :py:mod:`config class <clinicadl.optim.optimizers.config>`.

    See Also
    --------
    :py:class:`~clinicadl.models.SupervisedModel`
        For supervised training.
    """

    _config_type = ReconstructionModelConfig

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
