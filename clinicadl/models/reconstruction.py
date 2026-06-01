from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from .base import Model
from .vanilla import VanillaModel, VanillaModelConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch


class ReconstructionModelConfig(
    VanillaModelConfig, ObjectConfig["ReconstructionModel"]
):
    """
    Config class for ReconstructionModel.
    """

    @classmethod
    def _get_class(cls) -> type[Model]:
        """Returns the class associated to this config class."""
        return ReconstructionModel


class ReconstructionModel(VanillaModel, HasConfig[ReconstructionModelConfig]):
    """
    A vanilla reconstruction model, to work with simple AutoEncoders like
    :py:class:`~clinicadl.networks.nn.AutoEncoder`.

    Only the :py:meth:`forward_step` differs from :py:class:`~clinicadl.model.SupervisedModel`.

    Parameters
    ----------
    network : NetworkOrConfig
        The neural network, passed as a :py:class:`torch.nn.Module` or
        a :py:mod:`configuration object <clinicadl.networks.config>`.
    loss : LossOrConfig
        The reconstruction loss function, passed as a ``callable`` that returns a **1-item** :py:class:`~torch.Tensor`,
        or a :py:mod:`configuration object <clinicadl.losses.config>`.

        .. important::
            The loss function must have a :torch:`PyTorch style <nn.html#loss-functions>`,
            with an attribute named ``reduction`` that can be set to ``none``.

    optimizer : OptimizerConfig
        The optimizer, passed as a :py:mod:`configuration object <clinicadl.optim.optimizers.config>`.

    See Also
    --------
    :py:class:`~clinicadl.models.SupervisedModel`
        For supervised training.
    """

    config: ReconstructionModelConfig
    _config_type = ReconstructionModelConfig

    def forward_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs a pass forward in the neural network and a comparison with the input image.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.

        Returns
        -------
        torch.Tensor
            The computed loss, as a **1-item** :py:class:`torch.Tensor`.
        """
        images = batch.get_field("image", dtype=torch.float32)
        outputs = self.network(images)

        loss = self.loss(outputs, images)

        return loss
