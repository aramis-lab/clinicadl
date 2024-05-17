from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.preprocessing.preprocessing_config import PreprocessingConfig

logger = getLogger("clinicadl.utils.caps_dataset.data_config")


class DataConfig(BaseModel):  # TODO : put in data module
    """Config class to specify the data.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    label: Optional[str] = None
    label_code: Dict[str, int] = {}
    multi_cohort: bool = False
    n_proc: int = 1
    data_df: Optional[pd.DataFrame] = None
    transformations: Optional[Callable] = None
    label_presence: bool = False
    augmentation_transformations: Optional[Callable] = None
    multi_cohort: bool = False
    caps_dict: Optional[Dict] = None
    eval_mode: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator("diagnoses", mode="before")
    def validator_diagnoses(cls, v):
        """Transforms a list to a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check if columns are in tsv
