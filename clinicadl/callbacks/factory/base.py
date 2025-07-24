from abc import ABC
from typing import Any

from ..training_state import _TrainingState


class Callback(ABC):
    """
    Base class for defining training callbacks in ClinicaDL.

    Callbacks provide hooks into key events during the training loop, such as the start/end
    of training, epochs, batches, and validation phases. Subclass this class and override
    the desired methods to implement custom behavior (e.g., logging, early stopping,
    checkpointing, etc.).

    All methods receive a `_TrainingState` object and optional keyword arguments containing
    context-specific information.

    Examples
    --------
    Creating a custom callback:

    .. code-block:: python

        from clinicadl.callbacks import Callback

        class PrintLossCallback(Callback):
            def on_batch_end(self, config: _TrainingState, **kwargs):
                print(f"Loss: {config.current_loss:.4f}")

    Using callbacks in a training loop:

    .. code-block:: python

        callbacks = [PrintLossCallback(), EarlyStoppingCallback(patience=5)]
        trainer = Trainer(..., callbacks=callbacks)

        for split in splits:
            trainer.on_train_begin()

            for epoch in range(num_epochs):

                trainer.on_epoch_begin()

                for batch in train_loader:

                    trainer.on_batch_begin()

                    loss = trainer.training_step(batch)

                    trainer.on_backward_begin()
                    loss.backward()
                    trainer.on_backward_end()

                    optimizer.step()

                    trainer.on_batch_end()

                trainer.on_validation_begin()
                trainer.validate()
                trainer.on_validation_end()

                trainer.on_epoch_end()

            trainer.on_train_end(state)
    """

    def __init__(self):
        pass

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """Called once at the beginning of training."""
        pass

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """Called once at the end of training."""
        pass

    def on_epoch_begin(self, config: _TrainingState, **kwargs) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, config: _TrainingState, **kwargs) -> None:
        """Called before processing each training batch."""
        pass

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        """Called after processing each training batch."""
        pass

    def on_backward_begin(self, config: _TrainingState, **kwargs) -> None:
        """Called before the backward pass."""
        pass

    def on_backward_end(self, config: _TrainingState, **kwargs) -> None:
        """Called after the backward pass."""
        pass

    def on_validation_begin(self, config: _TrainingState, **kwargs) -> None:
        """Called before the validation loop starts."""
        pass

    def on_validation_end(self, config: _TrainingState, **kwargs) -> None:
        """Called after the validation loop ends."""
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
