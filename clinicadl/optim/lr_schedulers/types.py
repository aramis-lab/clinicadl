from typing import Union

from config import LRSchedulerConfig
from torch.optim.lr_scheduler import LRScheduler

LRSchedulerOrConfig = Union[LRScheduler, LRSchedulerConfig]
