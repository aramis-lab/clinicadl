from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.splitter.split_utils import find_splits
from clinicadl.splitter.validation import ValidationConfig

logger = getLogger("clinicadl.split_config")


class SplitConfig(BaseModel):
    """
    Abstract config class for the validation procedure.

    selection_metrics is specific to the task, thus it needs
    to be specified in a subclass.
    """

    n_splits: NonNegativeInt = 0
    split: Optional[Tuple[NonNegativeInt, ...]] = None
    tsv_path: Path  # not needed in predict ?

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("split", mode="before")
    def validator_split(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check that split exists (and check coherence with n_splits)

    def adapt_cross_val_with_maps_manager_info(
        self, maps_manager
    ):  # maps_manager is of type MapsManager but need to be in a MapsConfig type in the future
        # TEMPORARY
        if not self.split:
            self.split = find_splits(maps_manager.maps_path)
        logger.debug(f"List of splits {self.split}")


class SplitterConfig(BaseModel, ABC):
    """

    Abstract config class for the training pipeline.
    Some configurations are specific to the task (e.g. loss function),
    thus they need to be specified in a subclass.
    """

    data: DataConfig
    split: SplitConfig
    validation: ValidationConfig
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(
            data=kwargs,
            split=kwargs,
            validation=kwargs,
        )

    def _update(self, config_dict: Dict[str, Any]) -> None:
        """Updates the configs with a dict given by the user."""
        self.data.__dict__.update(config_dict)
        self.split.__dict__.update(config_dict)
        self.validation.__dict__.update(config_dict)
