import torch

from clinicadl.data.dataloader import Batch

from .base import BuiltinClinicaDLModel


class SupervisedModel(BuiltinClinicaDLModel):
    def training_step(self, batch: Batch) -> torch.Tensor:
        """
        Performs the training step on the model using the provided batch of data and returns
        the computed loss.

        It is on this loss that gradients will be computed.

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
        labels = batch.get_field("label")

        outputs = self.network(images)

        loss = self.loss(outputs, labels)

        return loss

    def evaluation_step(self, batch: Batch) -> Batch:
        """
        Performs the evaluation step where a validation batch is passed through
        the neural network and an output batch is inferred.

        The output batch contains :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`
        on which the :py:mod:`metrics <clinicadl.metrics` will be computed.

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
        images = batch.get_field("image", dtype=torch.float32)
        outputs = self.network(images)
        batch.add_field("output", outputs)

        return batch
