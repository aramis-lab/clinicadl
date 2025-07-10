from typing import Optional, Union

import torch

from clinicadl.data.dataloader import Batch

from .base import ClinicaDLModel


class Reconstruction(ClinicaDLModel):
    def training_step(
        self, data: Batch, device: Optional[Union[str, torch.device, int]] = None
    ) -> torch.Tensor:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """
        labels = data.get_labels().to(device)
        images = data.get_images().to(device)

        outputs = self.network(images)
        labels = labels.unsqueeze(dim=-1)

        loss = self.loss(outputs, labels)

        return loss

    def evaluation_step(
        self,
        data: Batch,
        device: Optional[Union[str, torch.device, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a training step on the model using the provided batch of data and return the computed loss
        """
        labels = data.get_labels().to(device)
        images = data.get_images().to(device)

        outputs = self.network(images)
        labels = labels.unsqueeze(dim=-1)

        return outputs, labels
