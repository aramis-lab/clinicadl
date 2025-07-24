from typing import Any, Optional, Union

import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.optim.lr_schedulers.config.base import LRSchedulerConfig
from clinicadl.optim.lr_schedulers.config.enum import ImplementedLRScheduler
from clinicadl.optim.lr_schedulers.config.factory import get_lr_scheduler_config

from .base import Callback

LRSchedulerType = Union[
    LRSchedulerConfig,
    ImplementedLRScheduler,
    torch.optim.lr_scheduler.LRScheduler,
    str,
]


class LRScheduler(Callback):
    """
    Learning Rate Scheduler Callback for training in ClinicaDL.

    This callback provides flexible integration of PyTorch learning rate schedulers
    into the training loop. It supports various input types for defining the scheduler,
    such as:

    - A string corresponding to a predefined scheduler name (e.g., ``"LinearLR"``)
    - A `ImplementedLRScheduler` enum value
    - A `LRSchedulerConfig` object with full custom configuration (**recommended for reproducibility**)
    - A `torch.optim.lr_scheduler.LRScheduler` instance directly

    It allows flexible definition and initialization of a scheduler at the beginning of training,
    and steps it after every batch to adjust the learning rate dynamically.

    Parameters
    ----------
    scheduler : Union[str, ImplementedLRScheduler, LRSchedulerConfig, torch.optim.lr_scheduler.LRScheduler]
        The learning rate scheduler configuration or object. Can be:

    **kwargs
        Additional keyword arguments passed to the scheduler config factory
        (only used when `scheduler` is a string or enum).

    Raises
    ------
    ValueError
        If the input type for `scheduler` is unsupported, or if the optimizer passed
        to the scheduler does not match the model's optimizer.

    Example
    -------
    Using a predefined name:

    .. code-block:: python

        from clinicadl.callbacks import LRScheduler
        scheduler = LRScheduler("LinearLR", start_factor=0.1, total_iters=10)

    Using a config object:

    .. code-block:: python

        from clinicadl.optim.lr_schedulers.config import LinearLRConfig
        scheduler = LRScheduler(LinearLRConfig(start_factor=0.1, total_iters=10))

    Using a PyTorch scheduler directly:

    .. code-block:: python

        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = LRScheduler(torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10))
    """

    def __init__(self, scheduler: LRSchedulerType, **kwargs):
        self.config: Optional[LRSchedulerConfig] = None
        self.torch_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        if isinstance(scheduler, str):
            scheduler = ImplementedLRScheduler(scheduler)

        if isinstance(scheduler, ImplementedLRScheduler):
            self.config = get_lr_scheduler_config(scheduler, **kwargs)

        elif isinstance(scheduler, LRSchedulerConfig):
            self.config = scheduler

        elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.torch_scheduler = scheduler
        else:
            raise ValueError(
                f"Invalid scheduler type: {type(scheduler)}. "
                f"Expected LRSchedulerConfig, ImplementedLRScheduler or torch.optim.lr_scheduler.LRScheduler"
            )

    def optimizers_equal(
        self, opt1: torch.optim.Optimizer, opt2: torch.optim.Optimizer
    ) -> bool:
        if type(opt1) != type(opt2):
            return False

        if len(opt1.param_groups) != len(opt2.param_groups):
            return False

        for g1, g2 in zip(opt1.param_groups, opt2.param_groups):
            # Comparer les hyperparamètres sauf les 'params' eux-mêmes
            for key in g1:
                if key == "params":
                    continue
                if g1[key] != g2.get(key):
                    return False
        return True

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Initialize the learning rate scheduler using the model's optimizer.

        Parameters
        ----------
        config : _TrainingState
            The training state, must include `model.optimizer`.
        """

        if not hasattr(config.model, "optimizer"):
            raise AttributeError("config.model must have an 'optimizer' attribute")

        optimizer = config.model.optimizer
        initial_lr = optimizer.param_groups[0].get("lr", None)

        if initial_lr is None:
            raise ValueError("Optimizer does not have a learning rate defined")

        # Initialize scheduler depending on configuration

        if self.config:
            self.scheduler = self.config.get_object(config.model.optimizer)

        elif self.torch_scheduler:
            if not (
                self.torch_scheduler.optimizer.defaults == optimizer.defaults
                and isinstance(self.torch_scheduler.optimizer, type(optimizer))
            ):
                raise ValueError(
                    f"The scheduler's optimizer you provided ({self.torch_scheduler.optimizer}) does not match "
                    f"the model's optimizer ({optimizer})."
                )
            self.scheduler = self.torch_scheduler

        else:
            raise ValueError("Scheduler not properly initialized.")

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Step the learning rate scheduler after each training batch.
        """

        if self.scheduler is None:
            raise RuntimeError(
                "Scheduler has not been initialized (call on_train_begin first)."
            )
        self.scheduler.step()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()

        if self.config:
            config_dict = self.config.to_dict()
            scheduler = config_dict.pop("name", None)
            json_dict.update({"scheduler": scheduler})
            json_dict.update(config_dict)

        if self.torch_scheduler:
            json_dict.update(self.torch_scheduler.__dict__)

        return json_dict
