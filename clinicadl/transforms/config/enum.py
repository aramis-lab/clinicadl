from clinicadl.utils.enum import BaseEnum


class ImplementedTransform(str, BaseEnum):
    """
    Implemented transforms in ClinicaDL.
    see: https://torchio.readthedocs.io/transforms/transforms.html
    """

    RESCALE_INTENSITY = "RescaleIntensity"
    Z_NORMALIZATION = "ZNormalization"
    MASK = "Mask"
    CROP_OR_PAD = "CropOrPad"
    TO_CANONICAL = "ToCanonical"
    CLAMP = "Clamp"
    RESAMPLE = "Resample"
    RESIZE = "Resize"
    ENSURE_MULTIPLE = "EnsureShapeMultiple"
    CROP = "Crop"
    PAD = "Pad"
    REMAP_LABELS = "RemapLabels"
    ONE_HOT = "OneHot"

    ONE_OF = "OneOf"

    RANDOM_FLIP = "RandomFlip"
    RANDOM_AFFINE = "RandomAffine"
    RANDOM_DEFORMATION = "RandomElasticDeformation"
    RANDOM_ANISOTROPY = "RandomAnisotropy"
    RANDOM_MOTION = "RandomMotion"
    RANDOM_GHOSTING = "RandomGhosting"
    RANDOM_SPIKE = "RandomSpike"
    RANDOM_BIAS_FIELD = "RandomBiasField"
    RANDOM_BLUR = "RandomBlur"
    RANDOM_NOISE = "RandomNoise"
    RANDOM_SWAP = "RandomSwap"
    RANDOM_GAMMA = "RandomGamma"

    ACTIVATIONS = "Activations"
    AS_DISCRETE = "AsDiscrete"
    KLCC = "KeepLargestConnectedComponent"
    EDT = "DistanceTransformEDT"
    SMALL_OBJECTS = "RemoveSmallObjects"
    LABEL_FILTER = "LabelFilter"
    FILL_HOLES = "FillHoles"
    SOBEL = "SobelGradients"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented transforms are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class AnatomicalLabel(str, BaseEnum):
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


class InterpolationMode(str, BaseEnum):
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


class EnsureShapeMultipleMode(str, BaseEnum):
    """
    Supported modes for TorchIO's EnsureShapeMultiple.
    see: https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.EnsureShapeMultiple
    """

    CROP = "crop"
    PAD = "pad"


class PaddingMode(str, BaseEnum):
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


class CenterMode(str, BaseEnum):
    """
    Supported options for the parameter 'center' in TorchIO's RandomAffine.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine
    """

    IMAGE = "image"
    ORIGIN = "origin"


class RandomAffinePaddingMode(str, BaseEnum):
    """
    Supported options for the parameter 'default_pad_value' in TorchIO's RandomAffine.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine
    """

    MINIMUM = "minimum"
    MEAN = "mean"
    OTSU = "otsu"


class AnatomicalAxis(str, BaseEnum):
    """
    Supported names for anatomical axes in TorchIO.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomFlip
    """

    LEFT_RIGHT = "LR"
    POSTERIOR_ANTERIOR = "PA"
    INFERIOR_SUPERIOR = "IS"


class NumericalAxis(int, BaseEnum):
    """
    Indexation of spatial axes in 3D.
    """

    ZERO = 0
    ONE = 1
    TWO = 2


class LockedBordersMode(int, BaseEnum):
    """
    Modes for 'locked_borders' argument in RandomElasticDeformation.
    see: https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomElasticDeformation
    """

    ZERO = 0
    ONE = 1
    TWO = 2


class Rounding(str, BaseEnum):
    """
    Options for 'rounding' argument in AsDiscrete.
    see: https://docs.monai.io/en/stable/transforms.html#monai.transforms.AsDiscrete
    """

    TORCH_ROUNDING = "torchrounding"


class SobelPaddingMode(str, BaseEnum):
    """
    Options for 'padding_mode' argument in SobelGradients.
    see: https://docs.monai.io/en/stable/transforms.html#monai.transforms.SobelGradients
    """

    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
