from enum import Enum


class ImplementedMetric(str, Enum):
    """Implemented metrics in ClinicaDL."""

    LOSS = "Loss"

    CONF_MATRIX = "ConfusionMatrixMetric"
    ROC_AUC = "ROCAUCMetric"

    MSE = "MSEMetric"
    MAE = "MAEMetric"
    RMSE = "RMSEMetric"
    PSNR = "PSNRMetric"
    SSIM = "SSIMMetric"
    MS_SSIM = "MultiScaleSSIMMetric"

    DICE = "DiceMetric"
    GENERALIZED_DICE = "GeneralizedDiceScore"
    IOU = "MeanIoU"
    SURF_DIST = "SurfaceDistanceMetric"
    HAUSDORFF = "HausdorffDistanceMetric"
    SURF_DICE = "SurfaceDiceMetric"

    # MMD = "MMDMetric"

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
