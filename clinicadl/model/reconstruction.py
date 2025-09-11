import torch

from clinicadl.data.dataloader import Batch

from .supervised import SupervisedModel


class ReconstructionModel(SupervisedModel):
    """
    A vanilla reconstruction model, to work with simple AutoEncoders like
    :py:class:`~clinicadl.networks.nn.AutoEncoder`.

    Only the :py:meth:`training_step` differs from :py:class:`~clinicadl.model.SupervisedModel`.

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
    :py:class:`clinicadl.model.SupervisedModel`
        A ``ClinicaDLModel`` for supervised training.
    """

    def training_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs a pass forward in the autoencoder and a comparison with the input image.

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
        images = batch.get_field("image", dtype=torch.float32)
        outputs = self.network(images)

        loss = self.loss(outputs, images)

        return loss
