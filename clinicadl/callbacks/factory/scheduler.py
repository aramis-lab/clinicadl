import torch

from clinicadl.train.training_state import _TrainingState

from .base import Callback


class Scheduler(Callback):
    """Base class for callbacks."""

    def __init__(self):
        self.scheduler = None

    def on_train_begin(self, config: _TrainingState, **kwargs):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            config.model.optimizer,
            max_lr=config.model.optimizer.param_groups[0]["lr"],
            steps_per_epoch=config.n_batch,
            epochs=config.optim.epochs,
        )

    def on_batch_end(self, config: _TrainingState, **kwargs):
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized")
        self.scheduler.step()
