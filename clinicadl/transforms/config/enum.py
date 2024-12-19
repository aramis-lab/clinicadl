from enum import Enum

from clinicadl.utils.enum import BaseEnum


class TransformType(str, BaseEnum):
    """
    Sources for transforms in ClinicaDL.
    """

    TORCHIO = "TorchIO"
    HOMEMADE = "HomeMade"


class ImplementedTransform(str, BaseEnum):
    """
    Implemented transforms in ClinicaDL.
    see: https://torchio.readthedocs.io/transforms/transforms.html
    """

    NAN_REMOVAL = "NanRemoval"

    RESCALE_INTENSITY = "RescaleIntensity"
    Z_NORMALIZATION = "ZNormalization"
    CLAMP = "Clamp"
    RESIZE = "Resize"
    ENSURE_MULTIPLE = "EnsureShapeMultiple"
    CROP = "Crop"
    PAD = "Pad"
    REMAP_LABELS = "RemapLabels"
    ONE_HOT = "OneHot"

    RANDOM_FLIP = "RandomFlip"
    RANDOM_AFFINE = "RandomAffine"
    RANDOM_DEFORMATION = "RandomElasticDeformation"
    RANDOM_ANISOTROPY = "RandomAnisotropy"
    RANDOM_MOTION = "RandomMotion"
    RANDOM_GHOSTING = "RandomGhosting"
    RANDOM_SPIKE = "RandomSpike"
    RANDOM_BIAS_FIELD = "RandomBiasField"
    RANDOM_BLUR = "RandomBlurConfig"
    RANDOM_NOISE = "RandomNoise"
    RANDOM_SWAP = "RandomSwap"
    RANDOM_GAMMA = "RandomGamma"


class AnatomicalLabel(str, Enum):
    """
    Anatomical regions provided by TorchIO.
    see: https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.preprocessing.intensity.NormalizationTransform
    """

    LEFT = "Left"
    RIGHT = "Right"
    POSTERIOR = "Posterior"
    ANTERIOR = "Anterior"
    INFERIOR = "Inferior"
    SUPERIOR = "Superior"


class InterpolationMode(str, Enum):
    """
    Supported interpolation modes in TorchIO.
    see: https://torchio.readthedocs.io/transforms/transforms.html#interpolation
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BSPLINE = "bspline"
    CUBIC = "cubic"
    GAUSSIAN = "gaussian"
    LABEL_GAUSSIAN = "label_gaussian"
    HAMMING = "hamming"
    COSINE = "cosine"
    WELCH = "welch"
    LANCZOS = "lanczos"
    BLACKMAN = "blackman"


class EnsureShapeMultipleMode(str, Enum):
    """
    Supported modes for TorchIO's EnsureShapeMultiple.
    see: https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.EnsureShapeMultiple
    """

    CROP = "crop"
    PAD = "pad"


class PaddingMode(str, Enum):
    """
    Supported padding modes for TorchIO's Pad.
    see: https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.Pad
    """

    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"


class CenterMode(str, Enum):
    """
    Supported options for the parameter 'center' in TorchIO's RandomAffine.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine
    """

    IMAGE = "image"
    ORIGIN = "origin"


class RandomAffinePaddingMode(str, Enum):
    """
    Supported options for the parameter 'default_pad_value' in TorchIO's RandomAffine.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine
    """

    MINIMUM = "minimum"
    MEAN = "mean"
    OTSU = "otsu"


class AnatomicalAxis(str, Enum):
    """
    Supported names for anatomical axes in TorchIO.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomFlip
    """

    LEFT_RIGHT = "LR"
    ANTERIOR_POSTERIOR = "AP"
    INFERIOR_SUPERIOR = "IS"


class NumericalAxis(str, Enum):
    """
    Indexation of spatial axes in 3D.
    """

    ZERO = 0
    ONE = 1
    TWO = 2


class LockedBordersMode(int, Enum):
    """
    Modes for 'locked_borders' argument in RandomElasticDeformation.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomElasticDeformation
    """

    ZERO = 0
    ONE = 1
    TWO = 2
