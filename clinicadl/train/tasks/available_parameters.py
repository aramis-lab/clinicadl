from enum import Enum


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


class Task(str, Enum):
    """Tasks that can be performed in ClinicaDL."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECONSTRUCTION = "reconstruction"


class Compensation(str, Enum):
    """Available compensations in ClinicaDL."""

    MEMORY = "memory"
    TIME = "time"


class SizeReductionFactor(int, Enum):
    """Available size reduction factors in ClinicaDL."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class ExperimentTracking(str, Enum):
    """Available tools for experiment tracking in ClinicaDL."""

    MLFLOW = "mlflow"
    WANDB = "wandb"


class Sampler(str, Enum):
    """Available samplers in ClinicaDL."""

    RANDOM = "random"
    WEIGHTED = "weighted"


class Mode(str, Enum):
    """Available modes in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    ROI = "roi"
    SLICE = "slice"
