from logging import getLogger
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.utils.enum import (
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.training_config")


class TransformsConfig(BaseModel):  # TODO : put in data module?
    """Config class to handle the transformations applied to th data."""

    data_augmentation: Tuple[Transform, ...] = ()
    train_transformations: Optional[Tuple[Transform, ...]] = None
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
