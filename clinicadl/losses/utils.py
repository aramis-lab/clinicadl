from typing import Callable, Tuple

from torch import Tensor

Loss = Callable[[*Tuple[Tensor, ...]], Tensor]
