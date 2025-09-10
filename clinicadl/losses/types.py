from typing import Callable, Union

from config import LossConfig
from torch import Tensor

Loss = Callable[..., Tensor]
LossOrConfig = Union[Loss, LossConfig]
