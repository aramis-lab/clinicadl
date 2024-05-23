from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

logger = getLogger("clinicadl.cross_validation_config")


class CrossValidationConfig(
    BaseModel
):  # TODO : put in data/cross-validation/splitter module
    """
    Config class to configure the cross validation procedure.

    tsv_directory is an argument that must be passed by the user.
    """

    n_splits: NonNegativeInt = 0
    split: Optional[Tuple[NonNegativeInt, ...]] = None
    tsv_directory: Optional[Path] = None  # not needed in predict ?
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("split", mode="before")
    def validator_split(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)
