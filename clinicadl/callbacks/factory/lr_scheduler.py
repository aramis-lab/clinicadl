from typing import Union

import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.optim.lr_schedulers.config.base import LRSchedulerConfig
from clinicadl.optim.lr_schedulers.config.enum import ImplementedLRScheduler
from clinicadl.optim.lr_schedulers.config.factory import get_lr_scheduler_config

from .base import Callback

LRSchedulerType = Union[
    LRSchedulerConfig, ImplementedLRScheduler, torch.optim.lr_scheduler.LRScheduler
]


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

    def __init__(self, scheduler: LRSchedulerType, **kwargs):
        self.scheduler = None

        if isinstance(scheduler, LRSchedulerConfig):
            self.config = scheduler
        elif isinstance(scheduler, ImplementedLRScheduler):
            self.config = get_lr_scheduler_config(scheduler, **kwargs)
        elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.config = None
            self.torch_scheduler = scheduler
        else:
            raise ValueError(
                f"Invalid scheduler type: {type(scheduler)}. "
                f"Expected LRSchedulerConfig, ImplementedLRScheduler or torch.optim.lr_scheduler.LRScheduler"
            )

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        if not hasattr(config.model, "optimizer"):
            raise AttributeError("config.model must have an 'optimizer' attribute")

        initial_lr = config.model.optimizer.param_groups[0].get("lr", None)
        if initial_lr is None:
            raise ValueError("Optimizer does not have a learning rate defined")

        if self.config:
            self.scheduler = self.config.get_object(config.model.optimizer)

        elif self.torch_scheduler:
            if self.torch_scheduler.optimizer is not config.model.optimizer:
                raise ValueError(
                    f"The scheduler's optimizer you provided ({self.torch_scheduler.optimizer}) does not match "
                    f"the model's optimizer ({config.model.optimizer})."
                )
            else:
                self.scheduler = self.torch_scheduler

        else:
            raise ValueError("LRScheduler has not been initialized.")

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        if self.scheduler is None:
            raise RuntimeError(
                "LRScheduler has not been initialized. Call on_train_begin first."
            )
        self.scheduler.step()
