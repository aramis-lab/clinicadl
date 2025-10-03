from abc import ABC
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clinicadl.data.dataloader import Batch, BatchType
from clinicadl.io import Maps
from clinicadl.models import ClinicaDLModel
from clinicadl.split import Split
from clinicadl.train import TrainerState


class Callback(ABC):
    """
    Base class for defining training callbacks in ClinicaDL.

    Callbacks provide hooks into key events during the training loop, such as the start/end
    of training, epochs, batches, and validation phases. Subclass this class and override
    the desired methods to implement custom behavior (e.g., logging, early stopping,
    checkpointing, etc.).

    All methods receive a `TrainerState` object and optional keyword arguments containing
    context-specific information.

    Callbacks allow you to define actions to perform
    such as logging, saving checkpoints, or early stopping.

    Some default callbacks are already included in the training loop, but you can also add
    your own by passing them to the :py:class:`~clinicadl.train.trainer.Trainer` via the ``callbacks`` argument:

    .. code-block:: python

        trainer = Trainer(..., callbacks=[MyCustomCallback(), AnotherCallback()])

    Each callback should inherit from the :py:class:`~clinicadl.callbacks.factory.base.Callback` class
    and implement the appropriate event methods, such as: ``on_train_start``, ``on_epoch_end``,
    ``on_validation_end``, etc.

    Callbacks should capture NON-ESSENTIAL logic

    This makes it easy to customize training behavior without modifying the core training logic.

    Examples
    --------
    Creating a custom callback:

    .. code-block:: python

        from clinicadl.callbacks import Callback

        class PrintLossCallback(Callback):
            def on_batch_end(self, config: TrainerState, **kwargs):
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
        """Initialize the callback."""

    # Train

    def on_train_begin(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState, split: Split
    ) -> None:
        """Called after the backward pass."""

    def on_train_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
    ) -> None:
        """Called after the backward pass."""

    def on_epoch_begin(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState
    ) -> None:
        """Called once at the beginning of training."""

    def on_epoch_end(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState
    ) -> None:
        """Called once at the beginning of training."""

    def on_forward_step_begin(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState, batch: BatchType
    ) -> None:
        """Called before processing each training batch."""

    def on_forward_step_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
        loss: torch.Tensor,
    ) -> None:
        """Called before processing each training batch."""

    def on_optimization_step_begin(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        loss: torch.Tensor,
    ) -> None:
        """Called before the backward pass."""

    def on_optimization_step_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler,
    ) -> None:
        """Called after the backward pass."""

    # Evaluate

    def on_evaluate_begin(
        self, model: ClinicaDLModel, maps: Maps, state: TrainerState, split: Split
    ) -> None:
        """Called after the backward pass."""

    def on_evaluate_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        metrics: pd.DataFrame,
        detailed_metrics: pd.DataFrame,
    ) -> None:
        """Called after the backward pass."""

    def on_evaluation_step_begin(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
    ) -> None:
        """"""

    def on_evaluation_step_end(
        self,
        model: ClinicaDLModel,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
        output: Batch,
        metrics: pd.DataFrame,
    ) -> None:
        """"""

    def state_dict(self, checkpoint_path: Path, **kwargs) -> None:
        """To save a checkpoint of the callback state."""

    def load_checkpoint(self, checkpoint_path: Path, **kwargs) -> None:
        """To load a checkpoint saved with 'save_checkpoint'."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        return {"name": self.__class__.__name__}


class Tracker(Callback):
    """Base class for defining experiment trackers in ClinicaDL."""

    def __init__(self):
        super().__init__()

        self.package = ""

    def is_available(self):
        """Check if the package is installed and available"""
        return find_spec(self.package) is not None
