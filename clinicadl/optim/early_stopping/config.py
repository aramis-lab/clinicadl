from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class EarlyStoppingConfig(BaseModel):
    """Config class to perform Early Stopping."""

    patience: Optional[PositiveInt] = None
    min_delta: NonNegativeFloat = 0.0
    mode: Mode = Mode.MIN
    check_finite: bool = True
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @model_validator(mode="after")
    def check_bounds(self) -> EarlyStoppingConfig:
        if self.upper_bound is not None and self.lower_bound is not None:
            assert (
                self.lower_bound < self.upper_bound
            ), "Upper bound should be greater than lower bound."
        return self
