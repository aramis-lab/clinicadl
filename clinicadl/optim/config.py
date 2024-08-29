from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveInt

from .early_stopping import EarlyStoppingConfig


class OptimizationConfig(BaseModel):
    """Config class to configure the optimization process."""

    accumulation_steps: PositiveInt = 1
    epochs: PositiveInt = 20
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    profiler: bool = False  # TODO : remove profiler. Not an optimization parameter
    # pydantic config
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
