import torch.nn as nn
from torch import Tensor


class NnWrapper(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        return self.network(x) + offset
