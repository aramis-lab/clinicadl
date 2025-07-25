from abc import ABC
from typing import Any

from .factory import *
from .factory.base import Callback
from .factory.checkpoint_saver import _CheckpointSaver
from .factory.logger import _Logger
from .factory.monitor import _Monitor
from .factory.training_loss import _TrainingLoss


def get_callback_from_dict(json_dict: dict[str, Any]) -> Callback:
    """
    Create a callback instance from a dictionary representation.

    Parameters
    ----------
    json_dict : dict
        Dictionary representation of the callback.

    Returns
    -------
    Callback
        Instantiated callback object.
    """
    callback_name = json_dict.pop("name")
    callback_class = globals()[callback_name]
    return callback_class(**json_dict)
