from typing import Union

from torch.optim import Optimizer

from .config import OptimizerConfig

OptimizerOrConfig = Union[Optimizer, OptimizerConfig]
