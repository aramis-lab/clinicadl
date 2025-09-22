from pathlib import Path
from typing import Any, Optional, Union

import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.optim.lr_schedulers.config import (
    ImplementedLRScheduler,
    LRSchedulerConfig,
)
from clinicadl.optim.lr_schedulers.config import LRSchedulerType as LRSchedulerMode
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
    optimizer : Optional[torch.optim.Optimizer], default=None
        The optimizer associated to the LR scheduler. **Mandatory if a name or a config class
        is passed to** ``scheduler``.
    scheduler_type : Optional[LRSchedulerMode], default=None
        The type of LR scheduler, among:

        - ``"epoch-based"``: learning rate is updated at the end of the epoch (e.g. :py:class:`~torch.optim.lr_scheduler.LinearLR`);
        - ``"loss-based"``: learning rate is updated at the end of the epoch according
          to the validation loss (e.g. :py:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`);
        - ``"step-based"``: learning rate is updated after each optimization step
          (e.g. :py:class:`~torch.optim.lr_scheduler.OneCycleLR`).

        **Mandatory if a raw LRScheduler is passed to** ``scheduler``. It will be ignore if a
        config class is passed.

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

    def __init__(
        self,
        scheduler: LRSchedulerType,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler_type: Optional[LRSchedulerMode] = None,
        **kwargs,
    ):
        self.config: Optional[LRSchedulerConfig] = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler
        self.scheduler_type: LRSchedulerType
        self._initial_state: dict

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.scheduler = scheduler
            if not scheduler_type:
                raise ValueError(
                    "If you pass directly your own LRScheduler, you must must specify the type of scheduler via 'scheduler_type'."
                )
            self.scheduler_type = LRSchedulerMode(scheduler_type)

        else:
            if isinstance(scheduler, str):
                scheduler = ImplementedLRScheduler(scheduler)

            if isinstance(scheduler, ImplementedLRScheduler):
                self.config = get_lr_scheduler_config(scheduler, **kwargs)

            elif isinstance(scheduler, LRSchedulerConfig):
                self.config = scheduler

            else:
                raise ValueError(
                    f"Invalid scheduler type: {type(scheduler)}. "
                    f"Expected LRSchedulerConfig, ImplementedLRScheduler or torch.optim.lr_scheduler.LRScheduler"
                )

            if not optimizer:
                raise ValueError(
                    "If you pass a LRScheduler via a name or a config class, you must also pass the associated optimizer via 'optimizer'."
                )

            self.scheduler = self.config.get_object(optimizer)
            self.scheduler_type = self.config.scheduler_type()

        self._initial_state = self.scheduler.state_dict()

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Reset the LR scheduler.
        """
        self.scheduler.load_state_dict(self._initial_state)

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Step the learning rate scheduler after each training batch for
        step-based schedulers.
        """
        if self.scheduler_type == LRSchedulerMode.STEP:
            self.scheduler.step()

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Step the learning rate scheduler after each epoch for
        epoch-based and loss-based schedulers.
        """
        if self.scheduler_type == LRSchedulerMode.EPOCH:
            self.scheduler.step()
        elif self.scheduler_type == LRSchedulerMode.LOSS:
            val_loss = config.metrics.get_loss(epoch=config.epoch)
            self.scheduler.step(val_loss)

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """To save the state of the LR scheduler."""
        state = self.scheduler.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        """To load a checkpoint saved with 'save_checkpoint'."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
        self.scheduler.load_state_dict(checkpoint)

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

        else:
            json_dict.update(self.scheduler.__dict__)

        return json_dict
