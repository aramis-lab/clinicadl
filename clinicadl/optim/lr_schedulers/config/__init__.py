"""Config classes for LR schedulers supported natively in ``ClinicaDL``. Based
on :torch:`PyTorch LR schedulers <optim.html#how-to-adjust-learning-rate>`."""

from .base import LRSchedulerConfig
from .configs import *
from .enum import ImplementedLRScheduler, LRSchedulerType
from .factory import get_lr_scheduler_config
