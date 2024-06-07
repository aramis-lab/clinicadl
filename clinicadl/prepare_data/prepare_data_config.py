from logging import getLogger
from pathlib import Path
from time import time
from typing import Annotated, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    ExtractionMethod,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SUVRReferenceRegions,
    Tracer,
)

logger = getLogger("clinicadl.predict_config")


class PrepareDataConfig(BaseModel):
    caps_directory: Path
    preprocessing_cls: Preprocessing
    n_proc: int = 1
    tsv_file: Optional[Path] = None
    extract_json: Optional[str] = None
    use_uncropped_image: bool = False
    tracer_cls: Tracer = Tracer.FFDG
    suvr_reference_region_cls: SUVRReferenceRegions = (
        SUVRReferenceRegions.CEREBELLUMPONS2
    )
    custom_suffix: str = ""
    dti_measure_cls: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space_cls: DTISpace = DTISpace.ALL
    save_features: bool = False
    extract_method: ExtractionMethod
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("extract_json", mode="before")
    def compute_extract_json(cls, v: str):
        if v is None:
            return f"extract_{int(time())}.json"
        elif not v.endswith(".json"):
            return f"{v}.json"
        else:
            return v

    @property
    def preprocessing(self) -> Preprocessing:
        return self.preprocessing_cls

    @preprocessing.setter
    def preprocessing(self, value: Union[str, Preprocessing]):
        self.preprocessing_cls = Preprocessing(value)

    @property
    def suvr_reference_region(self) -> SUVRReferenceRegions:
        return self.suvr_reference_region_cls

    @suvr_reference_region.setter
    def suvr_reference_region(self, value: Union[str, SUVRReferenceRegions]):
        self.suvr_reference_region_cls = SUVRReferenceRegions(value)

    @property
    def tracer(self) -> Tracer:
        return self.tracer_cls

    @tracer.setter
    def tracer(self, value: Union[str, Tracer]):
        self.tracer_cls = Tracer(value)

    @property
    def dti_measure(self) -> DTIMeasure:
        return self.dti_measure_cls

    @dti_measure.setter
    def dti_measure(self, value: Union[str, DTIMeasure]):
        self.dti_measure_cls = DTIMeasure(value)

    @property
    def dti_space(self) -> DTISpace:
        return self.dti_space_cls

    @dti_space.setter
    def dti_space(self, value: Union[str, DTISpace]):
        self.dti_space_cls = DTISpace(value)


class PrepareDataImageConfig(PrepareDataConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE


class PrepareDataPatchConfig(PrepareDataConfig):
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH


class PrepareDataSliceConfig(PrepareDataConfig):
    slice_direction_cls: SliceDirection = SliceDirection.SAGITTAL
    slice_mode_cls: SliceMode = SliceMode.RGB
    discarded_slices: Annotated[list[int], 2] = [0, 0]
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


class PrepareDataROIConfig(PrepareDataConfig):
    roi_list: list[str] = []
    roi_uncrop_output: bool = False
    roi_custom_template: str = ""
    roi_custom_mask_pattern: str = ""
    extract_method: ExtractionMethod = ExtractionMethod.ROI
