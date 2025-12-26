"""
To monitor and customize the training, evaluation, and prediction phases.
"""

from .base import Callback
from .implemented import (
    EarlyStoppingCallback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
)
