from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
from pydantic.dataclasses import dataclass


class PythaeModelOuput(ABC):
    loss: torch.Tensor


class PythaeModel(ABC):
    model_config: dataclass
    encoder: nn.Module
    decoder: nn.Module

    @abstractmethod
    def loss_function(
        self, recon_x: torch.Tensor, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass

    @abstractmethod
    def forward(self, inputs: OrderedDict, **kwargs) -> PythaeModelOuput:
        pass
