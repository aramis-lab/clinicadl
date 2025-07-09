from abc import ABC
from typing import Any

from .training_state import _TrainingState


class Callback(ABC):
    """Base class for callbacks."""

    def __init__(self):
        pass

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_epoch_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_batch_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_backward_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_backward_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_validation_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_validation_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = {"name": self.__class__.__name__}

        return json_dict
