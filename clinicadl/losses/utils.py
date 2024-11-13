from typing import Callable

from torch import Tensor

Loss = Callable[[Tensor, Tensor], Tensor]
