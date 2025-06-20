from enum import Enum


class ImageModality(str, Enum):
    """Possible modality for images in ClinicaDL."""

    T1W = "T1w"
    DWI = "dwi"
    PET = "pet"
    FLAIR = "FLAIR"
    CUSTOM = "custom"


class PreprocessingMethod(str, Enum):
    """Possible preprocessing methods available in Clinica."""

    T1_LINEAR = "t1-linear"
    PET_LINEAR = "pet-linear"
    FLAIR_LINEAR = "flair-linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"
