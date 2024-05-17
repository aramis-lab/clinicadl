from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from time import time
from typing import Annotated, Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
    SliceDirection,
    SliceMode,
)


class PreprocessingConfig(BaseModel):
    preprocessing_json: Optional[Path] = None
    preprocessing_cls: Preprocessing
    use_uncropped_image: bool = False
    extract_method: ExtractionMethod
    file_type: Optional[str] = None  # Optional ??
    save_features: bool = False
    extract_json: Optional[str] = None

    @property
    def preprocessing(self) -> Preprocessing:
        return self.preprocessing_cls

    @preprocessing.setter
    def preprocessing(self, value: Union[str, Preprocessing]):
        self.preprocessing_cls = Preprocessing(value)

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
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH


class PreprocessingSliceConfig(PreprocessingConfig):
    slice_direction_cls: SliceDirection = SliceDirection.SAGITTAL
    slice_mode_cls: SliceMode = SliceMode.RGB
    discarded_slices: Annotated[list[PositiveInt], 2] = [0, 0]
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    @property
    def slice_direction(self) -> SliceDirection:
        return self.slice_direction_cls

    @slice_direction.setter
    def slice_direction(self, value: Union[str, SliceDirection]):
        self.slice_direction_cls = SliceDirection(value)

    @property
    def slice_mode(self) -> SliceMode:
        return self.slice_mode_cls

    @slice_mode.setter
    def slice_mode(self, value: Union[str, SliceMode]):
        self.slice_mode_cls = SliceMode(value)


class PreprocessingROIConfig(PreprocessingConfig):
    roi_list: list[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_pattern: str = ""
    roi_custom_suffix: str = ""
    roi_custom_mask_pattern: str = ""
    roi_background_value: int = 0
    extract_method: ExtractionMethod = ExtractionMethod.ROI


def return_preprocessing_config(dict_: Dict[str, Any]):
    extract_method = ExtractionMethod(dict_["preprocessing"])
    if extract_method == ExtractionMethod.ROI:
        return PreprocessingROIConfig(**dict_)
    elif extract_method == ExtractionMethod.SLICE:
        return PreprocessingSliceConfig(**dict_)
    elif extract_method == ExtractionMethod.IMAGE:
        return PreprocessingImageConfig(**dict_)
    elif extract_method == ExtractionMethod.PATCH:
        return PreprocessingPatchConfig(**dict_)
