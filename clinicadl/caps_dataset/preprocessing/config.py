from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

logger = getLogger("clinicadl.modality_config")


class PreprocessingConfig(BaseModel):
    """
    Abstract config class for the validation procedure.

    """

    tsv_file: Optional[Path] = None
    preprocessing: Preprocessing

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class PETPreprocessingConfig(PreprocessingConfig):
    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2
    preprocessing: Preprocessing = Preprocessing.PET_LINEAR


class CustomPreprocessingConfig(PreprocessingConfig):
    custom_suffix: str = ""
    preprocessing: Preprocessing = Preprocessing.CUSTOM


class DTIPreprocessingConfig(PreprocessingConfig):
    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    preprocessing: Preprocessing = Preprocessing.DWI_DTI


class T1PreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T1_LINEAR


class FlairPreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.FLAIR_LINEAR


def return_mode_config(preprocessing: Preprocessing):
    if (
        preprocessing == Preprocessing.T1_EXTENSIVE
        or preprocessing == Preprocessing.T1_LINEAR
    ):
        return T1PreprocessingConfig
    elif preprocessing == Preprocessing.PET_LINEAR:
        return PETPreprocessingConfig
    elif preprocessing == Preprocessing.FLAIR_LINEAR:
        return FlairPreprocessingConfig
    elif preprocessing == Preprocessing.CUSTOM:
        return CustomPreprocessingConfig
    elif preprocessing == Preprocessing.DWI_DTI:
        return DTIPreprocessingConfig
    else:
        raise ValueError(f"Preprocessing {preprocessing.value} is not implemented.")
