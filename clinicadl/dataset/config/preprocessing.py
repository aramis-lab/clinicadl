import abc
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field

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


class PreprocessingConfig(BaseModel):
    """
    Abstract config class for the preprocessing procedure.
    """

    from_bids: bool = False
    preprocessing: Preprocessing
    use_uncropped_image: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @abc.abstractmethod
    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        pass

    @abc.abstractmethod
    def caps_nii(self) -> tuple:
        pass

    @abc.abstractmethod
    def get_filetype(self) -> FileType:
        pass

    def compute_folder(self, from_bids: bool = False) -> str:
        if from_bids:
            return self.preprocessing.value
        else:
            return self.preprocessing.value.replace("-", "_")

    @computed_field
    @property
    def file_type(self) -> FileType:
        if self.from_bids:
            file_type = self.bids_nii()

        elif self.preprocessing not in Preprocessing:
            raise NotImplementedError(
                f"Extraction of preprocessing {self.preprocessing.value} is not implemented from CAPS directory."
            )
        else:
            file_type = self.get_filetype()
        return file_type

    def linear_nii(self) -> FileType:
        needed_pipeline, modality = self.caps_nii()

        if self.use_uncropped_image:
            desc_crop = ""
        else:
            desc_crop = "_desc-Crop"

        file_type = FileType(
            pattern=f"*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality.value}.nii.gz",
            description=f"{modality.value} Image registered in MNI152NLin2009cSym space using {needed_pipeline.value} pipeline "
            + (
                ""
                if self.use_uncropped_image
                else "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            ),
            needed_pipeline=needed_pipeline,
        )
        return file_type


class PETPreprocessingConfig(PreprocessingConfig):
    tracer: Tracer = Tracer.FFDG
    suvr_reference_region: SUVRReferenceRegions = SUVRReferenceRegions.CEREBELLUMPONS2
    preprocessing: Preprocessing = Preprocessing.PET_LINEAR

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        trc = "" if self.tracer is None else f"_trc-{Tracer(self.tracer).value}"
        rec = "" if reconstruction is None else f"_rec-{reconstruction}"
        description = "PET data"

        if self.tracer:
            description += f" with {self.tracer.value} tracer"
        if reconstruction:
            description += f" and reconstruction method {reconstruction}"

        file_type = FileType(
            pattern=f"pet/*{trc}{rec}_pet.nii*", description=description
        )
        return file_type

    def caps_nii(self) -> tuple:
        return (self.preprocessing, ImageModality.PET)

    def get_filetype(self) -> FileType:
        if self.use_uncropped_image:
            description = ""
        else:
            description = "_desc-Crop"

        file_type = FileType(
            pattern=f"pet_linear/*_trc-{self.tracer.value}_space-MNI152NLin2009cSym{description}_res-1x1x1_suvr-{self.suvr_reference_region.value}_pet.nii.gz",
            description="",
            needed_pipeline="pet-linear",
        )
        return file_type


class CustomPreprocessingConfig(PreprocessingConfig):
    custom_suffix: str = ""
    preprocessing: Preprocessing = Preprocessing.CUSTOM

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(
            pattern=f"*{self.custom_suffix}",
            description="Custom suffix",
        )

    def caps_nii(self) -> tuple:
        return (self.preprocessing, ImageModality.CUSTOM)

    def get_filetype(self) -> FileType:
        return self.bids_nii()


class DTIPreprocessingConfig(PreprocessingConfig):
    dti_measure: DTIMeasure = DTIMeasure.FRACTIONAL_ANISOTROPY
    dti_space: DTISpace = DTISpace.ALL
    preprocessing: Preprocessing = Preprocessing.DWI_DTI

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="dwi/sub-*_ses-*_dwi.nii*", description="DWI NIfTI")

    def caps_nii(self) -> tuple:
        return (self.preprocessing, ImageModality.DWI)

    def get_filetype(self) -> FileType:
        """Return the query dict required to capture DWI DTI images.

        Parameters
        ----------
        config: DTIPreprocessingConfig

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


class T1PreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T1_LINEAR

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="anat/sub-*_ses-*_T1w.nii*", description="T1w MRI")

    def caps_nii(self) -> tuple:
        return (self.preprocessing, LinearModality.T1W)

    def get_filetype(self) -> FileType:
        return self.linear_nii()


class FlairPreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.FLAIR_LINEAR

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    def caps_nii(self) -> tuple:
        return (self.preprocessing, LinearModality.T2W)

    def get_filetype(self) -> FileType:
        return self.linear_nii()


class T2PreprocessingConfig(PreprocessingConfig):
    preprocessing: Preprocessing = Preprocessing.T2_LINEAR

    def bids_nii(self, reconstruction: Optional[str] = None) -> FileType:
        raise NotImplementedError(
            f"Extraction of preprocessing {self.preprocessing.value} is not implemented from BIDS directory."
        )

    def caps_nii(self) -> tuple:
        return (self.preprocessing, LinearModality.FLAIR)

    def get_filetype(self) -> FileType:
        return self.linear_nii()


ALL_PREPROCESSING_TYPES = Union[
    T1PreprocessingConfig,
    T2PreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
]
