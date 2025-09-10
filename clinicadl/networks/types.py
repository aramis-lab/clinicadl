from typing import Union

from torch.nn import Module

from .config import NetworkConfig

NetworkOrConfig = Union[Module, NetworkConfig]
