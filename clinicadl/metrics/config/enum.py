from enum import Enum


class ImplementedMetric(str, Enum):
    """Implemented metrics in ClinicaDL."""

    LOSS = "LossMetric"

    CONF_MATRIX = "ConfusionMatrixMetric"
    ROC_AUC = "ROCAUCMetric"
    AVERAGE_PRECISION = "AveragePrecisionMetric"

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


class ConfusionMatrixMetricName(str, Enum):
    """Supported metrics for ConfusionMatrixMetric."""

    SE = "sensitivity"
    RECALL = "recall"
    HIT_RATE = "hit_rate"
    TRUE_POSITIVE_RATE = "true_positive_rate"
    TPR = "tpr"

    SP = "specificity"
    SELECTIVITY = "selectivity"
    TRUE_NEGATIVE_RATE = "true_negative_rate"
    TNR = "tnr"

    P = "precision"
    POSITIVE_PREDICTIVE_VALUE = "positive_predictive_value"
    PPV = "ppv"

    NEGATIVE_PREDICTIVE_VALUE = "negative predictive value"
    NPV = "npv"

    MISS_RATE = "miss rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    FNR = "fnr"

    FALL_OUT = "fall out"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FPR = "fpr"

    FALSE_DISCOVERY_RATE = "false discovery rate"
    FDR = "fdr"

    FALSE_OMISSION_RATE = "false_omission_rate"
    FOR = "for"

    PREVALENCE_THRESHOLD = "prevalence_threshold"
    PT = "pt"

    THREAT_SCORE = "threat score"
    CRITICAL_SUCCESS_INDEX = "critical_success_index"
    TS = "ts"
    CSI = "csi"

    ACCURACY = "accuracy"
    ACC = "acc"

    BALANCED_ACCURACY = "balanced accuracy"
    BA = "ba"

    F1_SCORE = "f1 score"
    F1 = "f1"

    MATTHEWS_COEFF = "matthews correlation coefficient"
    MCC = "mcc"

    FOWLKES_MALLOWS = "fowlkes mallows index"
    FM = "fm"

    INFORMEDNESS = "informedness"
    BOOKMAKER = "bookmaker_informedness"
    BM = "bm"
    YOUDEN_INDEX = "youden_index"
    YOUDEN = "youden"

    MARKEDNESS = "markedness"
    DELTAP = "deltap"
    MK = "mk"


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
