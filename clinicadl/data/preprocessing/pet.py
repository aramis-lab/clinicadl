from logging import getLogger
from typing import Optional, Union

from pydantic import field_validator

from clinicadl.data.preprocessing.base import BasePreprocessing
from clinicadl.utils.enum import (
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.pet")


class PreprocessingPET(BasePreprocessing):
    """
    Configuration for PET image preprocessing
    """

    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2
    preprocessing: Preprocessing = Preprocessing.PET_LINEAR

    @field_validator("tracer", mode="before")
    def check_tracer(cls, v: Union[str, Tracer]):
        return Tracer(v)

    @field_validator("suvr_reference_region", mode="before")
    def check_suvr_reference_region(cls, v: Union[str, SUVRReferenceRegions]):
        return SUVRReferenceRegions(v)

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        trc, rec, description = "", "", "PET data"
        if self.tracer:
            description += f" with {self.tracer.value} tracer"
            trc = f"_trc-{self.tracer.value}"
        if reconstruction:
            description += f" and reconstruction method {reconstruction}"
            rec = f"_rec-{reconstruction}"

        return FileType(pattern=f"pet/*{trc}{rec}_pet.nii*", description=description)

    def get_caps_filetype(self) -> FileType:
        des_crop = "" if self.use_uncropped_image else "_desc-Crop"

        return FileType(
            pattern=f"pet_linear/*_trc-{self.tracer.value}_space-MNI152NLin2009cSym{des_crop}_res-1x1x1_suvr-{self.suvr_reference_region.value}_pet.nii.gz",
            description="",
            needed_pipeline="pet-linear",
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} PET images with tracer {self.tracer.value} and suvr reference region {self.suvr_reference_region.value}. "
