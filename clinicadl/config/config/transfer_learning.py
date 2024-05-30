from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, NonNegativeInt, field_validator

logger = getLogger("clinicadl.transfer_learning_config")


class TransferLearningConfig(BaseModel):
    """Config class to perform Transfer Learning."""

    nb_unfrozen_layer: NonNegativeInt = 0
    transfer_path: Optional[Path] = None
    transfer_selection_metric: str = "loss"
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("transfer_path", mode="before")
    def validator_transfer_path(cls, v):
        """Transforms a False to None."""
        if v is False:
            return None
        return v

    @field_validator("transfer_selection_metric")
    def validator_transfer_selection_metric(cls, v):
        return v  # TODO : check if metric is in transfer MAPS
