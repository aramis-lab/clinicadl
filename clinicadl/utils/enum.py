from enum import Enum


class Task(str, Enum):
    """Tasks that can be performed in ClinicaDL."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECONSTRUCTION = "reconstruction"


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


class ImageModality(str, Enum):
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


class SliceDirection(str, Enum):
    """Possible directions for a slice."""

    SAGITTAL = "0"
    CORONAL = "1"
    AXIAL = "2"


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


class Compensation(str, Enum):
    """Available compensations in ClinicaDL."""

    MEMORY = "memory"
    TIME = "time"


class ExperimentTracking(str, Enum):
    """Available tools for experiment tracking in ClinicaDL."""

    MLFLOW = "mlflow"
    WANDB = "wandb"


class Mode(str, Enum):
    """Available modes in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    ROI = "roi"
    SLICE = "slice"


class Optimizer(str, Enum):
    """Available optimizers in ClinicaDL."""

    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    ADAMW = "AdamW"
    ADAMAX = "Adamax"
    ASGD = "ASGD"
    NADAM = "NAdam"
    RADAM = "RAdam"
    RMSPROP = "RMSprop"
    SGD = "SGD"


class Sampler(str, Enum):
    """Available samplers in ClinicaDL."""

    RANDOM = "random"
    WEIGHTED = "weighted"


class SizeReductionFactor(int, Enum):
    """Available size reduction factors in ClinicaDL."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class Transform(str, Enum):  # TODO : put in transform module
    """Available transforms in ClinicaDL."""

    NOISE = "Noise"
    ERASING = "Erasing"
    CROPPAD = "CropPad"
    SMOOTHIN = "Smoothing"
    MOTION = "Motion"
    GHOSTING = "Ghosting"
    SPIKE = "Spike"
    BIASFIELD = "BiasField"
    RANDOMBLUR = "RandomBlur"
    RANDOMSWAP = "RandomSwap"


class ClassificationLoss(str, Enum):  # TODO : put in loss module
    """Available classification losses in ClinicaDL."""

    CrossEntropyLoss = "CrossEntropyLoss"
    MultiMarginLoss = "MultiMarginLoss"


class ClassificationMetric(str, Enum):  # TODO : put in metric module
    """Available classification metrics in ClinicaDL."""

    BA = "BA"
    ACCURACY = "accuracy"
    F1_SCORE = "F1_score"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    PPV = "PPV"
    NPV = "NPV"
    MCC = "MCC"
    MK = "MK"
    LR_PLUS = "LR_plus"
    LR_MINUS = "LR_minus"
    LOSS = "loss"


class ReconstructionLoss(str, Enum):  # TODO : put in loss module
    """Available reconstruction losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"
    VAEGaussianLoss = "VAEGaussianLoss"
    VAEBernoulliLoss = "VAEBernoulliLoss"
    VAEContinuousBernoulliLoss = "VAEContinuousBernoulliLoss"


class Normalization(str, Enum):  # TODO : put in model module
    """Available normalization layers in ClinicaDL."""

    BATCH = "BatchNorm"
    GROUP = "GroupNorm"
    INSTANCE = "InstanceNorm"


class Pooling(str, Enum):  # TODO : put in model module
    """Available pooling techniques in ClinicaDL."""

    MAXPOOLING = "MaxPooling"
    STRIDE = "stride"


class ReconstructionMetric(str, Enum):  # TODO : put in metric module
    """Available reconstruction metrics in ClinicaDL."""

    MAE = "MAE"
    RMSE = "RMSE"
    PSNR = "PSNR"
    SSIM = "SSIM"
    LOSS = "loss"


class RegressionLoss(str, Enum):  # TODO : put in loss module
    """Available regression losses in ClinicaDL."""

    L1Loss = "L1Loss"
    MSELoss = "MSELoss"
    KLDivLoss = "KLDivLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HuberLoss = "HuberLoss"
    SmoothL1Loss = "SmoothL1Loss"


class RegressionMetric(str, Enum):  # TODO : put in metric module
    """Available regression metrics in ClinicaDL."""

    R2_score = "R2_score"
    MAE = "MAE"
    RMSE = "RMSE"
    LOSS = "loss"


class GenerateType(str, Enum):
    """Available generation pipelines in ClinicaDL."""

    ART = "artifacts"
    TRI = "trivial"
    RAN = "random"
    SHE = "shepplogan"
    HYP = "hypometabolic"


class MaskChecksum(str, Enum):
    """Available hecksum pipelines in ClinicaDL."""

    AD = "2100d514a3fabab49fe30702700085a09cdad449bdf1aa04b8f804e238e4dfc2"
    BVFTD = "5a0ad28dff649c84761aa64f6e99da882141a56caa46675b8bf538a09fce4f81"
    LVPPA = "1099f5051c79d5b4fdae25226d97b0e92f958006f6545f498d4b600f3f8a422e"
    NFVPPA = "9512a4d4dc0003003c4c7526bf2d0ddbee65f1c79357f5819898453ef7271033"
    PCA = "ace36356b57f4db73e17c421a7cfd7ae056a1b258b8126534cf65d8d0be9527a"
    SVPPA = "44f2e00bf2d2d09b532cb53e3ba61d6087b4114768cc8ae3330ea84c4b7e0e6a"
    AAL2 = "89427970921674792481bffd2de095c8fbf49509d615e7e09e4bc6f0e0564471"


class MetricOptimumMin(str, Enum):
    MAE = "MAE"
    RMSE = "RMSE"
    LOSS = "loss"


class MetricOptimumMax(str, Enum):
    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    PPV = "PPV"
    NPV = "NPV"
    F1_SCORE = "F1_score"
    BA = "BA"
    PSNR = "PSNR"
    SSIM = "SSIM"
    LNCC = "LNCC"
