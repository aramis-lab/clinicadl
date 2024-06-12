from typing import Callable, Type, TypeVar

import torch
import torch.nn as nn

from clinicadl.network.pythae import ClinicaDLModel
from clinicadl.network.pythae.utils import PythaeModel

from .ae_config import AEConfig

T = TypeVar("T", bound="AE")


class CNN(ClinicaDLModel):
    def __init__(
        self,
        network: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.network = network
        self.loss = loss

    @classmethod
    def from_config(cls: Type[T], config: AEConfig) -> T:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(x)
        loss = self.compute_loss(y_pred, y)
        return loss

    def compute_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(y_pred, y)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
