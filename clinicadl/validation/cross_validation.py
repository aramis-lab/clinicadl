from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.splitter.split_utils import find_splits

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

    def adapt_cross_val_with_maps_manager_info(self, maps_manager: MapsManager):
        # TEMPORARY
        if not self.split:
            self.split = find_splits(maps_manager.maps_path, maps_manager.split_name)
        logger.debug(f"List of splits {self.split}")
