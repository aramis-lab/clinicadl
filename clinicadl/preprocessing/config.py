from logging import getLogger
from pathlib import Path
from time import time
from typing import Annotated, Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import PositiveInt

from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
    SliceDirection,
    SliceMode,
)

logger = getLogger("clinicadl.preprocessing_config")


class PreprocessingConfig(BaseModel):
    """
    Config class for preprocessing
    """

    preprocessing_json: Optional[Path] = None
    preprocessing: Preprocessing
    use_uncropped_image: bool = False
    extract_method: ExtractionMethod
    file_type: Optional[Dict[str, str]] = None  # Optional ??
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


class PreprocessingImageConfig(PreprocessingConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE


class PreprocessingPatchConfig(PreprocessingConfig):
    extract_method: ExtractionMethod = ExtractionMethod.PATCH
    patch_size: int = 50
    stride_size: int = 50


class PreprocessingSliceConfig(PreprocessingConfig):
    extract_method: ExtractionMethod = ExtractionMethod.SLICE
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    discarded_slices: Annotated[list[PositiveInt], 2] = [0, 0]


class PreprocessingROIConfig(PreprocessingConfig):
    extract_method: ExtractionMethod = ExtractionMethod.ROI
    roi_list: list[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
