from typing import Union

from config import NetworkConfig
from torch.nn import Module

NetworkOrConfig = Union[Module, NetworkConfig]
