from logging import getLogger
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
    SliceDirection,
    SliceMode,
)

logger = getLogger("clinicadl.preprocessing_config")


class ExtractionConfig(BaseModel):
    """
    Abstract config class for the validation procedure.


    """

    use_uncropped_image: bool = False
    extract_method: ExtractionMethod
    file_type: Optional[str] = None  # Optional ??
    save_features: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class ExtractionImageConfig(ExtractionConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE


class ExtractionPatchConfig(ExtractionConfig):
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH


class ExtractionSliceConfig(ExtractionConfig):
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    num_slices: Optional[NonNegativeInt] = None
    discarded_slices: Tuple[NonNegativeInt, NonNegativeInt] = (0, 0)
    extract_method: ExtractionMethod = ExtractionMethod.SLICE


class ExtractionROIConfig(ExtractionConfig):
    roi_list: list[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
    extract_method: ExtractionMethod = ExtractionMethod.ROI
