from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.preprocessing.preprocessing import read_preprocessing
from clinicadl.utils.enum import ExperimentTracking

logger = getLogger("clinicadl.callbacks_config")


class CallbacksConfig(BaseModel):
    """Config class to add callbacks to the training."""

    emissions_calculator: bool = False
    track_exp: Optional[ExperimentTracking] = None
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
