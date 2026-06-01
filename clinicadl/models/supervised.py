from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from clinicadl.infer import Inferer, SimpleInferer
from clinicadl.losses.types import LossOrConfig
from clinicadl.networks.types import NetworkOrConfig
from clinicadl.optim.optimizers.config import OptimizerConfig
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import IMAGE
from clinicadl.utils.objects import HasConfig

from .base import Model
from .vanilla import VanillaModel, VanillaModelConfig

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch


class SupervisedModelConfig(VanillaModelConfig, ObjectConfig["SupervisedModel"]):
    """
    Config class for SupervisedModel.
    """

    label_key: str

    @classmethod
    def _get_class(cls) -> type[Model]:
        """Returns the class associated to this config class."""
        return SupervisedModel


class SupervisedModel(VanillaModel, HasConfig[SupervisedModelConfig]):
    """
    A vanilla supervised model, for usual **classification**, **regression**,
    or **segmentation** task.

    Parameters
    ----------
    network : NetworkOrConfig
        The neural network, passed as a :py:class:`torch.nn.Module` or
        a :py:mod:`configuration object <clinicadl.networks.config>`.
    loss : LossOrConfig
        The loss function, passed as a ``callable`` that returns a **1-item** :py:class:`~torch.Tensor`,
        or a :py:mod:`configuration object <clinicadl.losses.config>`.

        .. important::
            The loss function must have a :torch:`PyTorch style <nn.html#loss-functions>`,
            with an attribute named ``reduction`` that can be set to ``none``.

    optimizer : OptimizerConfig
        The optimizer, passed as a :py:mod:`configuration object <clinicadl.optim.optimizers.config>`.

    label_key: str, default="label"
        The key of the label in the training :py:class:`samples <clinicadl.data.structures.Sample>`.

    See Also
    --------
    :py:class:`~clinicadl.models.ReconstructionModel`
        For image reconstruction.
    """

    config: SupervisedModelConfig
    _config_type = SupervisedModelConfig

    label_key: str

    def __init__(
        self,
        network: NetworkOrConfig,
        loss: LossOrConfig,
        optimizer: OptimizerConfig,
        inferer: Inferer = SimpleInferer(),
        label_key: str = "label",
    ):
        super().__init__(
            network=network,
            loss=loss,
            optimizer=optimizer,
            inferer=inferer,
            label_key=label_key,
        )
        self.label_key = self.config.label_key

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
        labels = batch.get_field(
            self.label_key, ensure_channel_dim=True, dtype=torch.float32
        )

        outputs = self.network(images)

        loss = self.loss(outputs, labels)

        return loss
