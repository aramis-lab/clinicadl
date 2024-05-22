from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.enum import (
    BIDSModality,
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

logger = getLogger("clinicadl.modality_config")


class ModalityConfig(BaseModel):
    """
    Abstract config class for the validation procedure.

    """

    tsv_file: Optional[Path] = None
    modality: BIDSModality

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)


class PETModalityConfig(ModalityConfig):
    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2
    modality: BIDSModality = BIDSModality.PET


class CustomModalityConfig(ModalityConfig):
    custom_suffix: str = ""
    modality: BIDSModality = BIDSModality.CUSTOM


class DTIModalityConfig(ModalityConfig):
    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    modality: BIDSModality = BIDSModality.DTI


class T1ModalityConfig(ModalityConfig):
    modality: BIDSModality = BIDSModality.T1


class FlairModalityConfig(ModalityConfig):
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
        raise ValueError(f"Preprocessing {preprocessing.value} is not implemented.")
