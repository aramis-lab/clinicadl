"""
For monitoring and customizing the training and evaluation phases.
"""

from .base import Callback
from .handler import CallbacksHandler
from .implemented import (
    EarlyStoppingCallback,
    LoggerCallback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
    MonitorCallback,
    TrainingCheckpointCallback,
)
