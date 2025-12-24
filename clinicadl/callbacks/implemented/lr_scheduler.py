from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

import pandas as pd
import torch
from pydantic import Field, field_validator, model_validator
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import Self

from clinicadl.optim.lr_schedulers.config import (
    LRSchedulerConfig,
    LRSchedulerType,
)
from clinicadl.optim.lr_schedulers.factory import get_lr_scheduler_from_dict
from clinicadl.optim.lr_schedulers.types import LRSchedulerOrConfig
from clinicadl.utils.config import ObjectConfig, ObjectOrConfig
from clinicadl.utils.dictionary.words import OPTIMIZER
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)
from clinicadl.utils.objects import HasConfig

from ..base import Callback
from .utils import get_metric_value

if TYPE_CHECKING:
    from clinicadl.train import TrainerState


class LRSchedulerCallbackConfig(ObjectConfig["LRSchedulerCallback"]):
    """Config class for ``LRSchedulerCallback``."""

    scheduler: ObjectOrConfig[LRScheduler, LRSchedulerConfig] = Field(
        reader=ObjectOrConfig.build_reader(get_lr_scheduler_from_dict)
    )
    optimizer_name: str
    scheduler_type: Optional[LRSchedulerType]
    metric_name: Optional[str]

    @field_validator("scheduler", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        """
        Converts a value to a ObjectOrConfig.
        """
        return ObjectOrConfig.from_value(v)

    @model_validator(mode="after")
    def _validate_scheduler_type(self) -> Self:
        """Checks that 'scheduler_type' is passed if necessary, otherwise retrieves it."""
        if isinstance(self.scheduler.value, LRScheduler) and not self.scheduler_type:
            raise ValueError(
                "If you pass directly your own LRScheduler, you must specify the type of scheduler via 'scheduler_type'."
            )
        elif isinstance(self.scheduler.value, LRSchedulerConfig):
            self.__dict__["scheduler_type"] = self.scheduler.value.scheduler_type()

        if (self.scheduler_type == LRSchedulerType.METRIC) and not self.metric_name:
            raise ValueError(
                f"If scheduler_type='{LRSchedulerType.METRIC.value}', you must "
                "pass the name of the validation metric via 'metric_name'."
            )

        return self

    @classmethod
    def _get_class(cls):
        return LRSchedulerCallback


class LRSchedulerCallback(Callback, HasConfig[LRSchedulerConfig]):
    """
    Learning Rate Scheduler to adjust the learning rate during optimization.

    Parameters
    ----------
    scheduler : Union[torch.optim.lr_scheduler.LRScheduler, LRSchedulerConfig]
        The learning rate scheduler passed as a raw :py:class:`torch.optim.lr_scheduler.LRScheduler` or via
        a :py:class:`~clinicadl.optim.lr_schedulers.config.LRSchedulerConfig`.
    optimizer_name : str, default="optimizer"
        The optimizer whose learning rate should be scheduled. It must be a name of one of the optimizers
        returned by the :py:meth:`Model.build_optimizers <clinicadl.models.Model.build_optimizers>`.
    scheduler_type : Optional[LRSchedulerMode], default=None
        The type of LR scheduler, among:

        - ``"epoch-based"``: learning rate is updated at the end of the epoch (e.g. :py:class:`~torch.optim.lr_scheduler.LinearLR`);
        - ``"metric-based"``: learning rate is updated at the end of the epoch according
          to a validation metric (e.g. :py:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`);
        - ``"step-based"``: learning rate is updated after each optimization step
          (e.g. :py:class:`~torch.optim.lr_scheduler.OneCycleLR`).

        **Mandatory if a raw LRScheduler is passed** to ``scheduler``. It will be ignore if a
        config class is passed.
    metric_name : Optional[str], default=None
        If ``scheduler_type="metric-based"``, it is the name of the metric to monitor.

    """

    _config_type = LRSchedulerCallbackConfig

    def __init__(
        self,
        scheduler: LRSchedulerOrConfig,
        optimizer_name: str = OPTIMIZER,
        scheduler_type: Optional[LRSchedulerType] = None,
        metric_name: Optional[str] = None,
    ):
        self.config: LRSchedulerCallbackConfig = self._config_type(
            scheduler=scheduler,
            optimizer_name=optimizer_name,
            scheduler_type=scheduler_type,
            metric_name=metric_name,
        )

        self.scheduler_config: Optional[LRSchedulerConfig] = None
        self.scheduler: Optional[LRScheduler] = None

        self._initial_state: Optional[dict] = None
        self._activated = False  # to prevent from calling in validation-only

        scheduler = self.config.scheduler.value
        if isinstance(scheduler, LRScheduler):
            self.scheduler = scheduler
            self._initial_state = self.scheduler.state_dict()
        elif isinstance(scheduler, LRSchedulerConfig):
            self.scheduler_config = scheduler

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(
        self,
        *,
        optimizers: dict[str, torch.optim.Optimizer],
        **kwargs,
    ) -> None:
        try:
            optimizer = optimizers[self.config.optimizer_name]
        except KeyError as exc:
            raise ClinicaDLArgumentError(
                f"In {type(self).__name__}, optimizer_name='{self.config.optimizer_name}' but there is no such optimizer (built with 'build_optimizers' method of your clinicadl.model.Model). "
                f"Optimizers are: {list(optimizers.keys())}"
            ) from exc

        if self.scheduler_config:
            self.scheduler = self.scheduler_config.get_object(optimizer)
        else:
            if optimizer is not self.scheduler.optimizer:
                raise ClinicaDLConfigurationError(
                    f"The optimizer associated to the LR scheduler {type(self.scheduler).__name__} is not the same as "
                    f"'{self.config.optimizer_name}' (returned by 'build_optimizers' method of your clinicadl.model.Model)."
                )
            self.scheduler.load_state_dict(self._initial_state)

        self._activated = True

    def on_optimization_step_end(self, **kwargs) -> None:
        if self.config.scheduler_type == LRSchedulerType.STEP:
            self.scheduler.step()

    def on_epoch_end(self, **kwargs) -> None:
        if self.config.scheduler_type == LRSchedulerType.EPOCH:
            self.scheduler.step()

    def on_validation_end(
        self, *, state: TrainerState, metrics_df: pd.DataFrame, **kwargs
    ) -> None:
        if self._activated and (self.config.scheduler_type == LRSchedulerType.METRIC):
            val_metric = get_metric_value(
                metrics_df,
                metric_name=self.config.metric_name,
                epoch=state.current_epoch,
            )
            self.scheduler.step(val_metric)

    def state_dict(self) -> Mapping[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)
