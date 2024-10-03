from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.splitter.split_utils import find_splits

logger = getLogger("clinicadl.validation_config")


class ValidationConfig(BaseModel):
    """
    Abstract config class for the validation procedure.

    selection_metrics is specific to the task, thus it needs
    to be specified in a subclass.
    """

    evaluation_steps: NonNegativeInt = 0
    selection_metrics: Tuple[str, ...] = ()
    valid_longitudinal: bool = False
    skip_leak_check: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("selection_metrics", mode="before")
    def validator_split(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)
