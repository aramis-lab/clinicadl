from logging import getLogger
from time import time
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.utils.clinica_utils import FileType
from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
    SliceMode,
)

logger = getLogger("clinicadl.preprocessing_config")


class ExtractionConfig(BaseModel):
    """
    Abstract config class for the Extraction procedure.
    """

    mode: ExtractionMethod
    file_type: Optional[FileType] = None
    save_features: bool = False
    extract_json: Optional[str] = None

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("extract_json", mode="before")
    def compute_extract_json(cls, v: str):
        if v is None:
            return f"extract_{int(time())}.json"
        elif not v.endswith(".json"):
            return f"{v}.json"
        else:
            return v


class ExtractionImageConfig(ExtractionConfig):
    mode: ExtractionMethod = ExtractionMethod.IMAGE


class ExtractionPatchConfig(ExtractionConfig):
    patch_size: int = 50
    stride_size: int = 50
    mode: ExtractionMethod = ExtractionMethod.PATCH


class ExtractionSliceConfig(ExtractionConfig):
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    num_slices: Optional[NonNegativeInt] = None
    discarded_slices: Tuple[NonNegativeInt, NonNegativeInt] = (0, 0)
    mode: ExtractionMethod = ExtractionMethod.SLICE


class ExtractionROIConfig(ExtractionConfig):
    roi_list: List[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
    mode: ExtractionMethod = ExtractionMethod.ROI
