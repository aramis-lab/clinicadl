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
    DTI = "dti"
    CUSTOM = "custom"


class LinearModality(str, Enum):
    T1W = "T1w"
    T2W = "T2w"
    FLAIR = "flair"


class Preprocessing(str, Enum):
    """Possible preprocessing method in clinicaDL."""

    T1_LINEAR = "t1-linear"
    T1_EXTENSIVE = "t1-extensive"
    PET_LINEAR = "pet-linear"
    FLAIR_LINEAR = "flair-linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"
    T2_LINEAR = "t2-linear"


class DTIMeasure(str, Enum):
    """Possible DTI measures."""

    FRACTIONAL_ANISOTROPY = "FA"
    MEAN_DIFFUSIVITY = "MD"
    AXIAL_DIFFUSIVITY = "AD"
    RADIAL_DIFFUSIVITY = "RD"


class DTISpace(str, Enum):
    """Possible DTI spaces."""

    NATIVE = "native"
    NORMALIZED = "normalized"
    ALL = "*"


class ExtractionMethod(str, Enum):
    """Possible extraction methods."""

    IMAGE = "image"
    SLICE = "slice"
    PATCH = "patch"
    ROI = "roi"


class SliceDirection(int, Enum):
    """Possible directions for a slice."""

    SAGITTAL = 0
    CORONAL = 1
    AXIAL = 2


class SliceMode(str, Enum):
    RGB = "rgb"
    SINGLE = "single"


class Template(str, Enum):
    T1_LINEAR = "MNI152NLin2009cSym"
    PET_LINEAR = "MNI152NLin2009cSym"
    FLAIR_LINEAR = "MNI152NLin2009cSym"


class Pattern(str, Enum):
    T1_LINEAR = ("res-1x1x1",)
    PET_LINEAR = ("res-1x1x1",)
    FLAIR_LINEAR = ("res-1x1x1",)
