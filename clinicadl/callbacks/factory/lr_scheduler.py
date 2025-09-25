from pathlib import Path
from typing import Any, Optional, Union

import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import PT
from clinicadl.optim.lr_schedulers.config import (
    ImplementedLRScheduler,
    LRSchedulerConfig,
)
from clinicadl.optim.lr_schedulers.config import LRSchedulerType as LRSchedulerMode
from clinicadl.optim.lr_schedulers.config.factory import get_lr_scheduler_config
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)

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
    optimizer : str, default="optimizer"
        The optimizer associated to the LR scheduler. **Mandatory if a name or a config class
        is passed to** ``scheduler``.
    scheduler_type : Optional[LRSchedulerMode], default=None
        The type of LR scheduler, among:

        - ``"epoch-based"``: learning rate is updated at the end of the epoch (e.g. :py:class:`~torch.optim.lr_scheduler.LinearLR`);
        - ``"metric-based"``: learning rate is updated at the end of the epoch according
          to a validation metric (e.g. :py:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`);
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
        optimizer_name: str = "optimizer",
        scheduler_type: Optional[LRSchedulerMode] = None,
        metric_name: Optional[str] = None,
        **kwargs,
    ):
        self.config: Optional[LRSchedulerConfig] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.optimizer_name = optimizer_name
        self.scheduler_type: LRSchedulerType
        self._initial_state: Optional[dict] = None

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.scheduler = scheduler
            if not scheduler_type:
                raise ValueError(
                    "If you pass directly your own LRScheduler, you must must specify the type of scheduler via 'scheduler_type'."
                )
            self.scheduler_type = LRSchedulerMode(scheduler_type)
            self._initial_state = self.scheduler.state_dict()

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

            self.scheduler_type = self.config.scheduler_type()

        if self.scheduler_type == LRSchedulerMode.METRIC and not metric_name:
            raise ClinicaDLConfigurationError(
                f"If scheduler_type='{LRSchedulerMode.METRIC.value}', you must "
                "pass the name of the validation metric via 'metric_name'."
            )
        self.metric_name = metric_name

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Checks the optimizer_name and instantiates the LR scheduler.
        """
        optimizers = config.model.get_optimizers()
        try:
            optimizer = optimizers[self.optimizer_name]
        except KeyError as exc:
            raise ClinicaDLArgumentError(
                f"In LRScheduler, optimizer_name='{self.optimizer_name}' but there is no such optimizer (returned by the 'get_optimizers' method of you ClinicaDLModel). "
                f"Optimizers are: {list(optimizers.keys())}"
            ) from exc

        if self.config:
            self.scheduler = self.config.get_object(optimizer)
        else:
            if optimizer is not self.scheduler.optimizer:
                raise ClinicaDLConfigurationError(
                    f"The optimizer associated to the LR scheduler '{type(self.scheduler).__name__}' is not the same as "
                    f"'{self.optimizer_name}' (returned by the 'get_optimizers' method of you ClinicaDLModel)."
                )
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
        epoch-based and metric-based schedulers.
        """
        if self.scheduler_type == LRSchedulerMode.EPOCH:
            self.scheduler.step()
        elif self.scheduler_type == LRSchedulerMode.METRIC:
            val_metric = config.metrics.get_metric(self.metric_name, epoch=config.epoch)
            self.scheduler.step(val_metric)

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """To save the state of the LR scheduler."""
        state = self.scheduler.state_dict()
        torch.save(state, checkpoint_path.with_suffix(PT))

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        """To load a checkpoint saved with 'save_checkpoint'."""
        checkpoint = torch.load(
            checkpoint_path.with_suffix(PT),
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
