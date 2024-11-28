import abc
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    ImageModality,
    LinearModality,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.modality_config")


class PreprocessingConfig(BaseModel, abc.ABC):
    """
    Abstract config class for the preprocessing procedure.
    """

    preprocessing: Preprocessing
    use_uncropped_image: bool = False

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def get_filetype(self, bids: bool = False) -> FileType:
        return self.get_bids_filetype() if bids else self.get_caps_filetype()

    @abc.abstractmethod
    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        """Abstract method to get the BIDS filetype."""
        pass

    @abc.abstractmethod
    def get_caps_filetype(self) -> FileType:
        """Abstract method to obtain FileType details."""
        pass

    @computed_field
    @property
    def file_type(self) -> FileType:
        if self.preprocessing not in Preprocessing:
            raise NotImplementedError(
                f"Extraction of preprocessing {self.preprocessing.value} is not implemented from CAPS directory."
            )
        else:
            return self.get_filetype()

    def linear_nii(
        self, modality: LinearModality, needed_pipeline: Preprocessing
    ) -> FileType:
        """
        Constructs the file type for linear caps image data
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"

        file_type = FileType(
            pattern=f"{self.preprocessing.value.replace('-', '_')}/*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality.value}.nii.gz",
            description=f"{modality.value} Image registered in MNI152NLin2009cSym space using {needed_pipeline.value} pipeline "
            + (
                ""
                if self.use_uncropped_image
                else "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            ),
            needed_pipeline=needed_pipeline,
        )
        return file_type


class PreprocessingPET(PreprocessingConfig):
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


class PreprocessingCustom(PreprocessingConfig):
    """
    Configuration for custom preprocessing with a user-defined suffix.
    """

    custom_suffix: str = ""
    preprocessing: Preprocessing = Preprocessing.CUSTOM

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(
            pattern=f"*{self.custom_suffix}",
            description="Custom suffix",
        )

    def get_caps_filetype(self) -> FileType:
        return FileType(
            pattern=f"custom/*{self.custom_suffix}",
            description="Custom suffix",
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} custom images with suffix {self.custom_suffix} "


class PreprocessingDTI(PreprocessingConfig):
    """
    Configuration for DTI-based preprocessing
    """

    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    preprocessing: Preprocessing = Preprocessing.DWI_DTI

    def get_bids_filerype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="dwi/sub-*_ses-*_dwi.nii*", description="DWI NIfTI")

    def get_caps_filetype(self) -> FileType:
        """Return the query dict required to capture DWI DTI images.

        Parameters
        ----------
        config: PreprocessingDTI

        Returns
        -------
        FileType :
        """
        measure = self.dti_measure
        space = self.dti_space

        return FileType(
            pattern=f"dwi/dti_based_processing/*/*_space-{space}_{measure.value}.nii.gz",
            description=f"DTI-based {measure.value} in space {space}.",
            needed_pipeline="dwi_dti",
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} DTI images with measure {self.dti_measure.value} and space {self.dti_space.value}. "


class PreprocessingT1(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T1_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="anat/sub-*_ses-*_T1w.nii*", description="T1w MRI")

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.T1W, needed_pipeline=Preprocessing.T1_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T1 images with t1-linear pipeline"


class PreprocessingFlair(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.FLAIR_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.FLAIR, needed_pipeline=Preprocessing.FLAIR_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} Flair images with flair-linear pipeline"


class PreprocessingT2(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T2_LINEAR

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        raise NotImplementedError(
            f"Extraction of preprocessing {self.preprocessing.value} is not implemented from BIDS directory."
        )

    def get_caps_filetype(self) -> FileType:
        return self.linear_nii(
            modality=LinearModality.T2W, needed_pipeline=Preprocessing.T2_LINEAR
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} T2 images with t2-linear pipeline"
