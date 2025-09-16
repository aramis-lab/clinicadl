from enum import Enum
from typing import Any, Optional, Union

import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.optim.lr_schedulers.config import (
    ImplementedLRScheduler,
    LRSchedulerConfig,
)
from clinicadl.optim.lr_schedulers.config.factory import get_lr_scheduler_config
from clinicadl.optim.optimizers.config import OptimizerConfig

from .base import Callback

LRSchedulerType = Union[
    LRSchedulerConfig,
    ImplementedLRScheduler,
    torch.optim.lr_scheduler.LRScheduler,
    str,
]


class LRSchedulerMode(str, Enum):
    STEP = "step-based"
    EPOCH = "epoch-based"
    METRIC = "metric"


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

    def __init__(
        self,
        scheduler: LRSchedulerType,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler_type: Optional[LRSchedulerType] = None,
        **kwargs,
    ):
        self.config: Optional[LRSchedulerConfig] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler_type: LRSchedulerType

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.scheduler = scheduler
            if type is None:
                raise ValueError(
                    "If you pass directly your own LRScheduler, you must must specify the type of scheduler via 'scheduler_type'."
                )
            self.scheduler_type = scheduler_type

        else:
            if not optimizer:
                raise ValueError(
                    "If you pass a LRScheduler via a name or a config class, you must also pass the associated optimizer via 'optimizer'."
                )

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

            self.scheduler_type = self._get_scheduler_type(scheduler.name)
            self.scheduler = scheduler.get_object(optimizer)

    def _check_optimizer_scheduler_consistency(
        self, optimizer: torch.optim.Optimizer, lr_scheduler_config: LRSchedulerConfig
    ) -> None:
        num_parameters = len(opt.param_groups)

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Step the learning rate scheduler after each training batch.
        """
        if self.scheduler_type == LRSchedulerMode.STEP:
            self.scheduler.step()

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Step the learning rate scheduler after each training batch.
        """
        if self.scheduler_type == LRSchedulerMode.EPOCH:
            self.scheduler.step()
        elif self.scheduler_type == LRSchedulerMode.METRIC:
            val_loss = config.metrics.get_loss(epoch=config.epoch)
            self.scheduler.step(val_loss)

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

    @staticmethod
    def _get_scheduler_type(scheduler: str) -> LRSchedulerType:
        if scheduler in {
            "ConstantLR",
            "ExponentialLR",
            "LinearLR",
            "StepLR",
            "MultiStepLR",
            "PolynomialLR",
            "",
        }:
            return LRSchedulerType.EPOCH
        elif scheduler in {"CyclicLR"}:
            return LRSchedulerType.STEP
        elif scheduler in {"ReduceLROnPlateau"}:
            return LRSchedulerType.METRIC
