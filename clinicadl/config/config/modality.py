from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.enum import (
    BIDSModality,
    DTIMeasure,
    DTISpace,
    SUVRReferenceRegions,
    Tracer,
)

logger = getLogger("clinicadl.modality_config")


class ModalityConfig(BaseModel):
    """
    Config class for modality.

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
