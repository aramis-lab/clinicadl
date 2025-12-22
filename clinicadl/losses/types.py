from typing import Callable, Union

from torch import Tensor

from .config import LossConfig

Loss = Callable[..., Tensor]
LossOrConfig = Union[Loss, LossConfig]
LossType = Union[Tensor, dict[str, Tensor]]
