from enum import Enum
from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .base import Callback
from .implemented import *


class ImplementedCallback(str, Enum):
    """Callbacks implemented natively in ClinicaDL."""

    EARLY_STOP = "EarlyStoppingCallback"
    LOGGER = "LoggerCallback"
    LR_SCHEDULER = "LRSchedulerCallback"
    MODEL_CHKPT = "ModelCheckpointCallback"
    MONITOR = "MonitorCallback"
    TRAIN_CHKPT = "TrainingCheckpointCallback"


@factory_from_dict(
    object_type=Callback,
    enum=ImplementedCallback,
    context=globals(),
    config=False,
)
def get_callback_from_dict(data: dict[str, Any], **kwargs) -> Callback:
    """
    Factory function to get a :py:class:`Callback` from a
    dictionary saved with :py:meth:`Callback.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    Callback
        The deserialized callback.
    """
