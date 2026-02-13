from copy import deepcopy

import monai.metrics as metrics
import monai.transforms
import pytest
import torchio as tio
from monai.metrics import ConfusionMatrixMetric
from pydantic import ValidationError

# pylint: disable=unused-import
from clinicadl.metrics.config import (
    ImplementedMetric,
    LossMetricConfig,
    MetricConfig,
)
from clinicadl.metrics.config.classification import (
    AveragePrecisionMetricConfig,
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.config.enum import ConfusionMatrixMetricName
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
from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper
from clinicadl.transforms.config import AsDiscreteConfig
from clinicadl.transforms.handlers import PostprocessingHandler
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

BAD_INPUTS = [
    ({"average": "abc"}, [ROCAUCMetricConfig, AveragePrecisionMetricConfig]),
    ({"metric_name": 0}, ConfusionMatrixMetricConfig),
    ({"compute_sample": ""}, ConfusionMatrixMetricConfig),
    (
        {"include_background": ""},
        [
            ConfusionMatrixMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            GeneralizedDiceScoreConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"class_thresholds": (0.1, 0), "include_background": ""}, SurfaceDiceMetricConfig),
    (
        {"reduction": "abc"},
        [
            ConfusionMatrixMetricConfig,
            MSEMetricConfig,
            MAEMetricConfig,
            RMSEMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            GeneralizedDiceScoreConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"max_val": 1, "reduction": "abc"}, PSNRMetricConfig),
    ({"class_thresholds": (0.1, 0), "reduction": "abc"}, SurfaceDiceMetricConfig),
    (
        {"spatial_dims": 2, "reduction": "abc"},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    (
        {"get_not_nans": True},
        [
            ConfusionMatrixMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"max_val": 1, "get_not_nans": True}, PSNRMetricConfig),
    ({"class_thresholds": (0.1, 0), "get_not_nans": True}, SurfaceDiceMetricConfig),
    (
        {"spatial_dims": 2, "get_not_nans": True},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    ({"max_val": 0}, PSNRMetricConfig),
    ({"spatial_dims": 1}, [SSIMMetricConfig, MultiScaleSSIMMetricConfig]),
    (
        {"spatial_dims": 2, "data_range": 0.0},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    (
        {"spatial_dims": 2, "kernel_type": "abc"},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    (
        {"spatial_dims": 2, "kernel_sigma": 0},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    ({"spatial_dims": 2, "k1": -0.1}, [SSIMMetricConfig, MultiScaleSSIMMetricConfig]),
    ({"spatial_dims": 2, "k2": -0.1}, [SSIMMetricConfig, MultiScaleSSIMMetricConfig]),
    ({"spatial_dims": 2, "win_size": 0}, SSIMMetricConfig),
    ({"spatial_dims": 2, "kernel_size": 0}, MultiScaleSSIMMetricConfig),
    (
        {"spatial_dims": 2, "weights": (0.1, -0.2)},
        MultiScaleSSIMMetricConfig,
    ),
    ({"ignore_empty": ""}, [DiceMetricConfig, MeanIoUConfig]),
    ({"num_classes": 0}, DiceMetricConfig),
    ({"return_with_label": True}, DiceMetricConfig),
    ({"weight_type": "abc"}, GeneralizedDiceScoreConfig),
    (
        {"distance_metric": "abc"},
        [
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"class_thresholds": (0.1, 0), "distance_metric": "abc"}, SurfaceDiceMetricConfig),
    ({"symmetric": ""}, SurfaceDistanceMetricConfig),
    ({"directed": ""}, HausdorffDistanceMetricConfig),
    ({"percentile": -0.1}, HausdorffDistanceMetricConfig),
    ({"class_thresholds": (0.1, 0), "use_subvoxels": ""}, SurfaceDiceMetricConfig),
]

GOOD_INPUTS = [
    ({"class_thresholds": (0.1, 0)}, SurfaceDiceMetricConfig),
    ({"average": "macro"}, [ROCAUCMetricConfig, AveragePrecisionMetricConfig]),
    ({"average": "micro"}, [ROCAUCMetricConfig, AveragePrecisionMetricConfig]),
    ({"average": "weighted"}, [ROCAUCMetricConfig, AveragePrecisionMetricConfig]),
    (
        {"metric_name": "sensitivity", "compute_sample": True},
        ConfusionMatrixMetricConfig,
    ),
    (
        {"include_background": True},
        [
            ConfusionMatrixMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            GeneralizedDiceScoreConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    (
        {"class_thresholds": (0.1, 0), "include_background": True},
        SurfaceDiceMetricConfig,
    ),
    (
        {"reduction": "sum"},
        [
            ConfusionMatrixMetricConfig,
            MSEMetricConfig,
            MAEMetricConfig,
            RMSEMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            GeneralizedDiceScoreConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"max_val": 1, "reduction": "sum"}, PSNRMetricConfig),
    ({"class_thresholds": (0.1, 0), "reduction": "sum"}, SurfaceDiceMetricConfig),
    (
        {"spatial_dims": 2, "reduction": "sum"},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    (
        {"reduction": "mean"},
        [
            ConfusionMatrixMetricConfig,
            MSEMetricConfig,
            MAEMetricConfig,
            RMSEMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            GeneralizedDiceScoreConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"max_val": 1, "reduction": "mean"}, PSNRMetricConfig),
    ({"class_thresholds": (0.1, 0), "reduction": "mean"}, SurfaceDiceMetricConfig),
    (
        {"spatial_dims": 2, "reduction": "mean"},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    (
        {"get_not_nans": False},
        [
            ConfusionMatrixMetricConfig,
            DiceMetricConfig,
            MeanIoUConfig,
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    ({"max_val": 1, "get_not_nans": False}, PSNRMetricConfig),
    ({"class_thresholds": (0.1, 0), "get_not_nans": False}, SurfaceDiceMetricConfig),
    (
        {"spatial_dims": 2, "get_not_nans": False},
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    ({"max_val": 1}, PSNRMetricConfig),
    (
        {
            "spatial_dims": 2,
            "data_range": 1.0,
            "kernel_type": "gaussian",
            "kernel_sigma": 0.1,
            "k1": 0,
            "k2": 0,
        },
        [SSIMMetricConfig, MultiScaleSSIMMetricConfig],
    ),
    ({"spatial_dims": 2, "kernel_type": "uniform", "win_size": 1}, SSIMMetricConfig),
    (
        {
            "spatial_dims": 2,
            "kernel_type": "uniform",
            "kernel_size": 1,
            "weights": (0.1, 0.2),
        },
        MultiScaleSSIMMetricConfig,
    ),
    ({"ignore_empty": True}, [DiceMetricConfig, MeanIoUConfig]),
    ({"num_classes": 2}, DiceMetricConfig),
    ({"num_classes": None, "return_with_label": False}, DiceMetricConfig),
    ({"weight_type": "square"}, GeneralizedDiceScoreConfig),
    ({"weight_type": "simple"}, GeneralizedDiceScoreConfig),
    ({"weight_type": "uniform"}, GeneralizedDiceScoreConfig),
    (
        {"distance_metric": "euclidean"},
        [
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    (
        {"class_thresholds": (0.1, 0), "distance_metric": "euclidean"},
        SurfaceDiceMetricConfig,
    ),
    (
        {"distance_metric": "taxicab"},
        [
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    (
        {"class_thresholds": (0.1, 0), "distance_metric": "taxicab"},
        SurfaceDiceMetricConfig,
    ),
    (
        {"distance_metric": "chessboard"},
        [
            SurfaceDistanceMetricConfig,
            HausdorffDistanceMetricConfig,
        ],
    ),
    (
        {"class_thresholds": (0.1, 0), "distance_metric": "chessboard"},
        SurfaceDiceMetricConfig,
    ),
    ({"symmetric": True}, SurfaceDistanceMetricConfig),
    ({"directed": True, "percentile": 0}, HausdorffDistanceMetricConfig),
    ({"percentile": None}, HausdorffDistanceMetricConfig),
    ({"class_thresholds": (0.1, 0), "use_subvoxels": True}, SurfaceDiceMetricConfig),
]


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize("args,configs", GOOD_INPUTS)
def test_good_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        c = config(**args)
        for arg, value in args.items():
            assert getattr(c, arg) == value


def test_confusion_matrix_metric():
    for metric in ConfusionMatrixMetricName:
        c = ConfusionMatrixMetricConfig(metric_name=metric)
        assert isinstance(c.get_object(), MonaiMetricWrapper)
        assert isinstance(c.get_object().metric, ConfusionMatrixMetric)


def test_check_spatial_dim():
    with pytest.raises(ValidationError):
        SSIMMetricConfig(win_size=(1, 2, 3), spatial_dims=2)
    with pytest.raises(ValidationError):
        MultiScaleSSIMMetricConfig(kernel_size=(2, 3), spatial_dims=3)


MANDATORY_ARGS = {
    "max_val": 1,
    "class_thresholds": (0.5, 0.5),
    "spatial_dims": 2,
    "postprocessing": [tio.Crop(cropping=1), AsDiscreteConfig(threshold=0.5)],
}


@pytest.mark.parametrize(
    "config,expected_class",
    [
        (ConfusionMatrixMetricConfig, metrics.ConfusionMatrixMetric),
        (ROCAUCMetricConfig, metrics.ROCAUCMetric),
        (AveragePrecisionMetricConfig, metrics.AveragePrecisionMetric),
        (MultiScaleSSIMMetricConfig, metrics.MultiScaleSSIMMetric),
        (PSNRMetricConfig, metrics.PSNRMetric),
        (SSIMMetricConfig, metrics.SSIMMetric),
        (MAEMetricConfig, metrics.MAEMetric),
        (MSEMetricConfig, metrics.MSEMetric),
        (RMSEMetricConfig, metrics.RMSEMetric),
        (DiceMetricConfig, metrics.DiceMetric),
        (GeneralizedDiceScoreConfig, metrics.GeneralizedDiceScore),
        (HausdorffDistanceMetricConfig, metrics.HausdorffDistanceMetric),
        (MeanIoUConfig, metrics.MeanIoU),
        (SurfaceDiceMetricConfig, metrics.SurfaceDiceMetric),
        (SurfaceDistanceMetricConfig, metrics.SurfaceDistanceMetric),
    ],
)
def test_get_object(config, expected_class):
    c: MetricConfig = config(**MANDATORY_ARGS)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, MonaiMetricWrapper)
    assert isinstance(transform_from_config.metric, expected_class)
    assert transform_from_config.pred_key == "output"
    assert transform_from_config.label_key == "label"
    assert len(transform_from_config.postprocessing.transforms) == 2
    assert isinstance(transform_from_config.postprocessing.transforms[0], tio.Crop)
    assert isinstance(
        transform_from_config.postprocessing.transforms[1].transform,
        monai.transforms.AsDiscrete,
    )

    c.label_key = None
    c.pred_key = "abc"
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config.postprocessing.transforms[0], tio.Crop)
    assert isinstance(
        transform_from_config.postprocessing.transforms[1],
        MonaiTransformWrapper,
    )
    assert transform_from_config.label_key is None
    assert transform_from_config.pred_key == "abc"

    # postprocessing
    new_args = deepcopy(MANDATORY_ARGS)
    new_args["postprocessing"] = PostprocessingHandler(MANDATORY_ARGS["postprocessing"])
    c: MetricConfig = config(**MANDATORY_ARGS)
    transform_from_config = c.get_object()
    assert len(transform_from_config.postprocessing.transforms) == 2


def test_name():
    for name in ImplementedMetric:
        config = globals()[f"{name.value}Config"]
        c = config(**MANDATORY_ARGS)
        assert c.name == name.value


@pytest.mark.parametrize(
    "config,optimum",
    [
        (ROCAUCMetricConfig, "max"),
        (AveragePrecisionMetricConfig, "max"),
        (MultiScaleSSIMMetricConfig, "max"),
        (PSNRMetricConfig, "max"),
        (SSIMMetricConfig, "max"),
        (MAEMetricConfig, "min"),
        (MSEMetricConfig, "min"),
        (RMSEMetricConfig, "min"),
        (DiceMetricConfig, "max"),
        (GeneralizedDiceScoreConfig, "max"),
        (HausdorffDistanceMetricConfig, "min"),
        (MeanIoUConfig, "max"),
        (SurfaceDiceMetricConfig, "max"),
        (SurfaceDistanceMetricConfig, "min"),
    ],
)
def test_optimum(config, optimum):
    assert config.optimum() == optimum


def test_optimum_confusion_matrix():
    config = ConfusionMatrixMetricConfig(metric_name="fpr")
    assert config.optimum() == "min"
    config = ConfusionMatrixMetricConfig(metric_name="tpr")
    assert config.optimum() == "max"
    metric = config.get_object()
    assert metric.optimum == "max"
