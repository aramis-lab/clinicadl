from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from .early_stopping import EarlyStoppingConfig
from .lr_scheduler import LRSchedulerConfig
from .optimizer import OptimizerConfig


class OptimizationConfig(BaseModel):
    """Config class to configure the optimization process."""

    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    profiler: bool = False  # TODO : remove profiler. Not an optimization parameter
    # pydantic config
    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    @model_validator(mode="after")
    def model_validator(self) -> OptimizationConfig:
        """
        Checks if parameter groups mentioned for the optimizer are mentioned
        for the LR scheduler.
        """
        if isinstance(self.lr_scheduler.min_lr, dict):
            scheduler_keys = set(self.lr_scheduler.min_lr.keys())
            optimizer_keys = self.optimizer.get_all_groups()
            for key in optimizer_keys:
                assert (
                    key in scheduler_keys
                ), f"You mentioned the parameter group '{key}' for the optimizer, so you must also pass a min_lr for this group."

        return self
