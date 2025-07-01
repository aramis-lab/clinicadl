from abc import ABC

from clinicadl.callbacks.training_state import _TrainingState


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
