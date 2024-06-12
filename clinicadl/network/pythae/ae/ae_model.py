from typing import Callable, Type, TypeVar

import torch
import torch.nn as nn

from clinicadl.network.pythae import ClinicaDLModel
from clinicadl.network.pythae.utils import PythaeModel

from .ae_config import AEConfig
from .ae_utils import PythaeAEWrapper

T = TypeVar("T", bound="AE")


class AE(ClinicaDLModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstruction_loss: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.MSELoss(),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

    @classmethod
    def from_config(cls: Type[T], config: AEConfig) -> T:
        pass

    @staticmethod
    def from_pythae(model: PythaeModel) -> PythaeAEWrapper:
        return PythaeAEWrapper(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.embed(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        recon_x = self.forward(x)
        loss = self.compute_loss(recon_x, x)
        return loss

    def compute_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_loss(recon_x, x)
