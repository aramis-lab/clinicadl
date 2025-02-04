from clinicadl.utils.enum import BaseEnum


class PreprocessingMethod(str, BaseEnum):
    """Preprocessing methods supported in ClinicaDL."""

    T1_LINEAR = "t1-linear"
    PET_LINEAR = "pet-linear"
    FLAIR_LINEAR = "flair-linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"
