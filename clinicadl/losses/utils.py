from typing import Callable, Union

from torch import Tensor
from torch.nn.modules.loss import _Loss

Loss = Union[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    _Loss,
]
