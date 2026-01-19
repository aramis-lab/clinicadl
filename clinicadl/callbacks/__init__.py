"""
To monitor and customize the training, evaluation, and prediction phases.
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
