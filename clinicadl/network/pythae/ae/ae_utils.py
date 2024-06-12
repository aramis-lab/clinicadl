from enum import Enum

import torch

from clinicadl.network.pythae import ClinicaDLModel
from clinicadl.network.pythae.utils import PythaeModel


class AENetworks(str, Enum):
    AE_Conv5_FC3 = "AE_Conv5_FC3"
    AE_Conv4_FC3 = "AE_Conv4_FC3"
    CAE_half = "CAE_half"


class PythaeAEWrapper(ClinicaDLModel):
    def __init__(self, model: PythaeModel):
        super().__init__()
        self.pythae_model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.embed(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.pythae_model.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.pythae_model.decoder(x)

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        inputs = {"data": x}
        loss = self.pythae_model.forward(inputs).loss
        return loss

    def compute_loss(
        self, recon_x: torch.Tensor, x: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        loss = self.pythae_model.loss_function(recon_x, x, **kwargs)
        if isinstance(loss, tuple):
            return loss[0]
        return loss
