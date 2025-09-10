from typing import Union

from torch.optim.lr_scheduler import LRScheduler

from .config import LRSchedulerConfig

LRSchedulerOrConfig = Union[LRScheduler, LRSchedulerConfig]
