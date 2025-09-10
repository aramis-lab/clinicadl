from typing import Union

from config import OptimizerConfig
from torch.optim import Optimizer

OptimizerOrConfig = Union[Optimizer, OptimizerConfig]
