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

    n_splits: NonNegativeInt = 0
    split: Optional[Tuple[NonNegativeInt, ...]] = None
    tsv_path: Optional[Path] = None  # not needed in predict ?
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("split", "selection_metrics", mode="before")
    def validator_split(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)

    def adapt_cross_val_with_maps_manager_info(
        self, maps_manager
    ):  # maps_manager is of type MapsManager but need to be in a MapsConfig type in the future
        # TEMPORARY
        if not self.split:
            self.split = find_splits(maps_manager.maps_path, maps_manager.split_name)
        logger.debug(f"List of splits {self.split}")
