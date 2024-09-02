from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary


class ImplementedLRScheduler(str, Enum):
    """Implemented LR schedulers in ClinicaDL."""

    CONSTANT = "ConstantLR"
    LINEAR = "LinearLR"
    STEP = "StepLR"
    MULTI_STEP = "MultiStepLR"
    PLATEAU = "ReduceLROnPlateau"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented LR schedulers are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Mode(str, Enum):
    """Supported mode for ReduceLROnPlateau."""

    MIN = "min"
    MAX = "max"


class ThresholdMode(str, Enum):
    """Supported threshold mode for ReduceLROnPlateau."""

    ABS = "abs"
    REL = "rel"


class LRSchedulerConfig(BaseModel):
    """Config class to configure the optimizer."""

    scheduler: Optional[ImplementedLRScheduler] = None
    step_size: Optional[PositiveInt] = None
    gamma: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    milestones: Optional[List[PositiveInt]] = None
    factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    start_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    end_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES

    mode: Union[Mode, DefaultFromLibrary] = DefaultFromLibrary.YES
    patience: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold_mode: Union[ThresholdMode, DefaultFromLibrary] = DefaultFromLibrary.YES
    cooldown: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    min_lr: Union[
        NonNegativeFloat, Dict[str, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    eps: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @field_validator("last_epoch")
    @classmethod
    def validator_last_epoch(cls, v):
        if isinstance(v, int):
            assert (
                -1 <= v
            ), f"last_epoch must be -1 or a non-negative int but it has been set to {v}."
        return v

    @field_validator("milestones")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        if v is not None:
            assert len(np.unique(v)) == len(
                v
            ), "Epoch(s) in milestones should be unique."
            return sorted(v)
        return v

    @model_validator(mode="after")
    def check_mandatory_args(self) -> LRSchedulerConfig:
        if (
            self.scheduler == ImplementedLRScheduler.MULTI_STEP
            and self.milestones is None
        ):
            raise ValueError(
                """If you chose MultiStepLR as LR scheduler, you should pass milestones
                (see PyTorch documentation for more details)."""
            )
        elif self.scheduler == ImplementedLRScheduler.STEP and self.step_size is None:
            raise ValueError(
                """If you chose StepLR as LR scheduler, you should pass a step_size
                (see PyTorch documentation for more details)."""
            )
        return self
