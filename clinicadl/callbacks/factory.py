from abc import ABC
from typing import Any

from .base import Callback
from .implemented import *

# from .implemented.logger import _Logger
# from .implemented.monitor import _Monitor
# from .implemented.training_loss import _TrainingLoss


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
