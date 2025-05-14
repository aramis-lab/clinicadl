from enum import Enum


class ClassificationMetrics(Enum):
    """
    Enum for classification metrics.
    """

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PPV = "ppv"
    NPV = "npv"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"


class RegressionMetrics(Enum):
    """
    Enum for regression metrics.
    """

    MSE = "mse"
    MAE = "mae"
    LOSS = "loss"


class ReconstructionMetrics(Enum):
    """
    Enum for reconstruction metrics.
    """

    PSNR = "psnr"
    SSIM = "ssim"
    MSE = "mse"
    MAE = "mae"
    LOSS = "loss"
