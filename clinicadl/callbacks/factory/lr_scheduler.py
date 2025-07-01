import torch

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback


class LRScheduler(Callback):
    """
    Learning rate scheduler callback using PyTorch's OneCycleLR.

    This callback initializes a OneCycleLR scheduler at the beginning of training
    and updates the learning rate after each batch.

    Usage
    -----
    Must be used with a PyTorch optimizer accessible via `config.model.optimizer`.
    The scheduler uses the optimizer's initial learning rate as the max_lr parameter.

    Events
    ------
    - on_train_begin: Initializes the OneCycleLR scheduler.
    - on_batch_end: Steps the scheduler to update learning rate.
    """

    def __init__(self):
        self.scheduler = None

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        if not hasattr(config.model, "optimizer"):
            raise AttributeError("config.model must have an 'optimizer' attribute")

        initial_lr = config.model.optimizer.param_groups[0].get("lr", None)
        if initial_lr is None:
            raise ValueError("Optimizer does not have a learning rate defined")

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=config.model.optimizer,
            max_lr=initial_lr,
            steps_per_epoch=config.n_batch,
            epochs=config.optim.epochs,
        )

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        if self.scheduler is None:
            raise RuntimeError(
                "LRScheduler has not been initialized. Call on_train_begin first."
            )
        self.scheduler.step()
