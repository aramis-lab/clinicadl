from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.caps_dataset.data_config import DataConfig
from clinicadl.utils.preprocessing.preprocessing import read_preprocessing

from .available_parameters import (
    Compensation,
    ExperimentTracking,
    Mode,
    Optimizer,
    Sampler,
    SizeReductionFactor,
    Transform,
)


class TransformsConfig(BaseModel):  # TODO : put in data module?
    """Config class to handle the transformations applied to th data."""

    data_augmentation: Tuple[Transform, ...] = ()
    normalize: bool = True
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("data_augmentation", mode="before")
    def validator_data_augmentation(cls, v):
        """Transforms lists to tuples and False to empty tuple."""
        if isinstance(v, list):
            return tuple(v)
        if v is False:
            return ()
        return v
