from enum import Enum


class ImplementedMetrics(str, Enum):
    """Implemented metrics in ClinicaDL."""

    LOSS = "Loss"

    RECALL = "Recall"
    SPECIFICITY = "Specificity"
    PRECISION = "Precision"
    NPV = "Negative Predictive Value"
    F1 = "F1 score"
    BALANCED_ACC = "Balanced Accuracy"
    ACC = "Accuracy"
    MARKEDNESS = "Markedness"
    MCC = "Matthews Correlation Coefficient"
    ROC_AUC = "ROCAUC"

    MSE = "MSE"
    MAE = "MAE"
    RMSE = "RMSE"
    PSNR = "PSNR"
    SSIM = "SSIM"
    MS_SSIM = "Multi-scale SSIM"

    DICE = "Dice"
    GENERALIZED_DICE = "Generalized Dice"
    IOU = "IoU"
    SURF_DIST = "Surface distance"
    HAUSDORFF = "Hausdorff distance"
    SURF_DICE = "Surface Dice"

    MMD = "MMD"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented metrics are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Reduction(str, Enum):
    """Supported reduction for the metrics."""

    MEAN = "mean"
    SUM = "sum"


class GeneralizedDiceScoreReduction(str, Enum):
    """Supported reduction for GeneralizedDiceScore."""

    MEAN = "mean_batch"
    SUM = "sum_batch"


class Average(str, Enum):
    """Supported averaging method for ROCAUCMetric."""

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"


class ConfusionMatrixMetric(str, Enum):
    """Supported metrics related to confusion matrix (in the format accepted by MONAI)."""

    RECALL = "recall"
    SPECIFICITY = "specificity"
    PRECISION = "precision"
    NPV = "negative predictive value"
    F1 = "f1 score"
    BALANCED_ACC = "balanced accuracy"
    ACC = "accuracy"
    MARKEDNESS = "markedness"
    MCC = "matthews correlation coefficient"


class DistanceMetric(str, Enum):
    "Supported distances."

    L2 = "euclidean"
    L1 = "taxicab"
    LINF = "chessboard"


class Kernel(str, Enum):
    "Supported kernel for SSIMMetric."

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class WeightType(str, Enum):
    "Supported weight types for GeneralizedDiceScore."

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"
