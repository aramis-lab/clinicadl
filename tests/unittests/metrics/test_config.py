from copy import deepcopy

import pytest
from pydantic import ValidationError
from torch.nn import MSELoss

from clinicadl.metrics.config import ImplementedMetric, create_metric_config
from clinicadl.metrics.config.base import LossMetricConfig
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.config.reconstruction import (
    MultiScaleSSIMMetricConfig,
    PSNRMetricConfig,
    SSIMMetricConfig,
)
from clinicadl.metrics.config.regression import (
    MAEMetricConfig,
    MSEMetricConfig,
    RMSEMetricConfig,
)
from clinicadl.metrics.config.segmentation import (
    DiceMetricConfig,
    GeneralizedDiceScoreConfig,
    HausdorffDistanceMetricConfig,
    MeanIoUConfig,
    SurfaceDiceMetricConfig,
    SurfaceDistanceMetricConfig,
)

MANDATORY_FIELDS = {
    "class_thresholds": (0.1, 0),
    "max_val": 1,
    "spatial_dims": 2,
    "loss_fn": lambda x: x,
}
BAD_INPUTS = [
    ("average", "abc"),
    ("metric_name", 0),
    ("compute_sample", ""),
    ("include_background", ""),
    ("reduction", "abc"),
    ("get_not_nans", True),
    ("max_val", 0),
    ("spatial_dims", 0),
    ("data_range", 0.0),
    ("kernel_type", "abc"),
    ("kernel_sigma", 0),
    ("k1", -0.1),
    ("k2", -0.1),
    ("win_size", 0),
    ("kernel_size", 0),
    ("weights", (0.1, -0.2)),
    ("ignore_empty", ""),
    ("num_classes", 0),
    ("return_with_label", True),
    ("weight_type", "abc"),
    ("distance_metric", "abc"),
    ("symmetric", ""),
    ("directed", ""),
    ("percentile", -0.1),
    ("class_thresholds", (0.1, -0.01)),
    ("use_subvoxels", ""),
    ("loss_fn", None),
    ("generalized_dice_reduction", None),
]

GOOD_INPUTS = [
    ("class_thresholds", (0.1, 0)),
    ("average", "macro"),
    ("metric_name", "recall"),
    ("compute_sample", True),
    ("include_background", True),
    ("reduction", "sum"),
    ("get_not_nans", False),
    ("data_range", 1.0),
    ("kernel_type", "gaussian"),
    ("kernel_sigma", 0.1),
    ("k1", 0),
    ("k2", 0),
    ("win_size", 1),
    ("kernel_size", 1),
    ("weights", (0.1, 0.2)),
    ("ignore_empty", True),
    ("num_classes", 2),
    ("return_with_label", False),
    ("generalized_dice_reduction", "mean_batch"),
    ("weight_type", "square"),
    ("distance_metric", "euclidean"),
    ("symmetric", True),
    ("directed", True),
    ("percentile", 0),
    ("use_subvoxels", True),
    ("average", "micro"),
    ("compute_sample", False),
    ("include_background", False),
    ("reduction", "mean"),
    ("kernel_type", "uniform"),
    ("ignore_empty", False),
    ("num_classes", None),
    ("generalized_dice_reduction", "sum_batch"),
    ("weight_type", "simple"),
    ("distance_metric", "chessboard"),
    ("symmetric", False),
    ("directed", False),
    ("percentile", None),
    ("use_subvoxels", False),
    ("average", "weighted"),
    ("weight_type", "uniform"),
    ("distance_metric", "taxicab"),
]


@pytest.mark.parametrize("arg,value", BAD_INPUTS)
def test_validation_fail(arg, value):
    for metric in ImplementedMetric:
        config = create_metric_config(metric)
        fields = config.model_fields
        if arg == "generalized_dice_reduction" and metric == "GeneralizedDiceScore":
            arg_ = "reduction"
        elif arg == "reduction" and metric == "GeneralizedDiceScore":
            continue
        else:
            arg_ = arg

        if arg_ in fields:
            mandatory_inputs = deepcopy(MANDATORY_FIELDS)
            if arg_ in mandatory_inputs:
                del mandatory_inputs[arg_]

            with pytest.raises(ValidationError):
                config(**{arg_: value}, **mandatory_inputs)


@pytest.mark.parametrize(
    "arg,value",
    GOOD_INPUTS,
)
def test_validation_pass(arg, value):
    for metric in ImplementedMetric:
        config = create_metric_config(metric)
        fields = config.model_fields
        if arg == "generalized_dice_reduction" and metric == "GeneralizedDiceScore":
            arg_ = "reduction"
        elif arg == "reduction" and metric == "GeneralizedDiceScore":
            continue
        else:
            arg_ = arg

        if arg_ in fields:
            mandatory_inputs = deepcopy(MANDATORY_FIELDS)
            if arg_ in mandatory_inputs:
                del mandatory_inputs[arg_]

            c = config(**{arg_: value}, **mandatory_inputs)
            assert getattr(c, arg_) == value


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("Loss", LossMetricConfig),
        ("ConfusionMatrixMetric", ConfusionMatrixMetricConfig),
        ("ROCAUCMetric", ROCAUCMetricConfig),
        ("MultiScaleSSIMMetric", MultiScaleSSIMMetricConfig),
        ("PSNRMetric", PSNRMetricConfig),
        ("SSIMMetric", SSIMMetricConfig),
        ("MAEMetric", MAEMetricConfig),
        ("MSEMetric", MSEMetricConfig),
        ("RMSEMetric", RMSEMetricConfig),
        ("DiceMetric", DiceMetricConfig),
        ("GeneralizedDiceScore", GeneralizedDiceScoreConfig),
        ("HausdorffDistanceMetric", HausdorffDistanceMetricConfig),
        ("MeanIoU", MeanIoUConfig),
        ("SurfaceDiceMetric", SurfaceDiceMetricConfig),
        ("SurfaceDistanceMetric", SurfaceDistanceMetricConfig),
    ],
)
def test_create_optimizer_config(name, expected_class):
    config = create_metric_config(name)
    assert config == expected_class


def test_check_spatial_dim():
    with pytest.raises(ValidationError):
        SSIMMetricConfig(win_size=(1, 2, 3), spatial_dims=2)
    with pytest.raises(ValidationError):
        MultiScaleSSIMMetricConfig(kernel_size=(2, 3), spatial_dims=3)


def test_check_reduction():
    with pytest.raises(ValidationError):
        LossMetricConfig(loss_fn=lambda x: x, reduction=None)
    config = LossMetricConfig(loss_fn=MSELoss(reduction="sum"), reduction=None)
    assert config.reduction == "sum"
