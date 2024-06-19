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
    Abstract config class for the preprocessing procedure.
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


class T2PreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T2_LINEAR
