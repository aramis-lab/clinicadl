from enum import Enum


class ClinicaDLEnum(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class InterpretationMethod(ClinicaDLEnum):
    """Possible interpretation method in clinicaDL."""

    GRADIENTS = "gradients"
    GRAD_CAM = "grad-cam"


class Preprocessing(ClinicaDLEnum):
    """Possible preprocessing method in clinicaDL."""

    T1_LINEAR = "t1-linear"
    T1_EXTENSIVE = "t1-extensive"
    PET_LINEAR = "pet-linear"
    FLAIR_LINEAR = "flair-linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"


class SUVRReferenceRegions(ClinicaDLEnum):
    """Possible SUVR reference region for pet images in clinicaDL."""

    PONS = "pons"
    CEREBELLUMPONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUMPONS2 = "cerebellumPons2"


class Tracer(ClinicaDLEnum):
    """Possible tracer for pet images in clinicaDL."""

    FFDG = "18FFDG"
    FAV45 = "18FAV45"


class Pathology(ClinicaDLEnum):
    """Possible pathology for hypometabolic generation of pet images in clinicaDL."""

    AD = "ad"
    BVFTD = "bvftd"
    LVPPA = "lvppa"
    NFVPPA = "nfvppa"
    PCA = "pca"
    SVPPA = "svppa"


class Modality(ClinicaDLEnum):
    """Possible tracer for pet images in clinicaDL."""

    T1 = "t1"
    DWI = "dwi"
    PET = "pet"
    FLAIR = "flair"
    T2 = "t2"
    DTI = "dwi-dti"
    CUSTOM = "custom"
