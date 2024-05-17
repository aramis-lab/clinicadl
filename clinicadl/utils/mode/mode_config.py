from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from clinicadl.utils.enum import (
    BIDSModality,
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SUVRReferenceRegions,
    Tracer,
)


class ModeConfig(BaseModel):
    tsv_file: Optional[Path] = None
    modality: BIDSModality


class PETModalityConfig(ModeConfig):
    tracer_cls: Tracer = Tracer.FFDG
    suvr_reference_region_cls: SUVRReferenceRegions = (
        SUVRReferenceRegions.CEREBELLUMPONS2
    )
    modality: BIDSModality = BIDSModality.PET

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


class CustomModalityConfig(ModeConfig):
    custom_suffix: str = ""
    modality: BIDSModality = BIDSModality.CUSTOM


class DTIModalityConfig(ModeConfig):
    dti_measure_cls: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space_cls: DTISpace = DTISpace.ALL
    modality: BIDSModality = BIDSModality.DTI

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


class T1ModalityConfig(ModeConfig):
    modality: BIDSModality = BIDSModality.T1


class FlairModalityConfig(ModeConfig):
    modality: BIDSModality = BIDSModality.FLAIR


def return_mode_config(preprocessing: Preprocessing):
    if (
        preprocessing == Preprocessing.T1_EXTENSIVE
        or preprocessing == Preprocessing.T1_LINEAR
    ):
        return T1ModalityConfig
    elif preprocessing == Preprocessing.PET_LINEAR:
        return PETModalityConfig
    elif preprocessing == Preprocessing.FLAIR_LINEAR:
        return FlairModalityConfig
    elif preprocessing == Preprocessing.CUSTOM:
        return CustomModalityConfig
    elif preprocessing == Preprocessing.DWI_DTI:
        return DTIModalityConfig
    else:
        raise ValueError(f"Preprocessing {preprocessing} is not implemented.")

        # custom_suffix: Optional[str] = None,
        # tracer_cls: Optional[Tracer] = None,
        # suvr_reference_region_cls: Optional[SUVRReferenceRegions] = None,
        # dti_measure_cls: Optional[DTIMeasure] = None,
        # dti_space_cls: Optional[DTISpace] = None,
        # tsv_file: Optional[Path] = None,
        # extract_json: Optional[str] = None,
