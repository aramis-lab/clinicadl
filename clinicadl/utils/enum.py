from enum import Enum


class InterpretationMethod(str, Enum):
    """Possible interpretation method in clinicaDL."""

    GRADIENTS = "gradients"
    GRAD_CAM = "grad-cam"


class SUVRReferenceRegions(str, Enum):
    """Possible SUVR reference region for pet images in clinicaDL."""

    PONS = "pons"
    CEREBELLUMPONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUMPONS2 = "cerebellumPons2"


class Tracer(str, Enum):
    """Possible tracer for pet images in clinicaDL."""

    FFDG = "18FFDG"
    FAV45 = "18FAV45"
    CPIB = "11CPIB"


class Pathology(str, Enum):
    """Possible pathology for hypometabolic generation of pet images in clinicaDL."""

    AD = "ad"
    BVFTD = "bvftd"
    LVPPA = "lvppa"
    NFVPPA = "nfvppa"
    PCA = "pca"
    SVPPA = "svppa"


class BIDSModality(str, Enum):
    """Possible modality for images in clinicaDL."""

    T1 = "t1"
    DWI = "dwi"
    PET = "pet"
    FLAIR = "flair"
    # T2 = "t2"
    DTI = "dti"
    CUSTOM = "custom"


class LinearModality(str, Enum):
    T1W = "T1w"
    T2W = "T2w"
    FLAIR = "flair"


class Preprocessing(str, Enum):
    """Possible preprocessing method in clinicaDL."""

    T1_LINEAR = "t1_linear"
    T1_EXTENSIVE = "t1_extensive"
    PET_LINEAR = "pet_linear"
    FLAIR_LINEAR = "flair_linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"
    T2_LINEAR = "t2_linear"


class DTIBasedMeasure(str, Enum):
    """Possible DTI measures."""

    FRACTIONAL_ANISOTROPY = "FA"
    MEAN_DIFFUSIVITY = "MD"
    AXIAL_DIFFUSIVITY = "AD"
    RADIAL_DIFFUSIVITY = "RD"
