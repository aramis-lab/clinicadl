import pytest
from pydantic import ValidationError

from clinicadl.metrics.config.segmentation import (
    DiceMetricConfig,
    GeneralizedDiceScoreConfig,
    HausdorffDistanceMetricConfig,
    MeanIoUConfig,
    SurfaceDiceMetricConfig,
    SurfaceDistanceMetricConfig,
)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"class_thresholds": [0.1], "reduction": "abc"},
        {"class_thresholds": [0.1], "get_not_nans": True},
    ],
)
def test_fails_validation(bad_inputs):
    with pytest.raises(ValidationError):
        DiceMetricConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        MeanIoUConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        SurfaceDistanceMetricConfig(**bad_inputs)


def test_fails_validation_dice():
    with pytest.raises(ValidationError):
        DiceMetricConfig(return_with_label=True)
    with pytest.raises(ValidationError):
        DiceMetricConfig(num_classes=0)


def test_fails_validation_gen_dice():
    with pytest.raises(ValidationError):
        GeneralizedDiceScoreConfig(reduction="mean")
    with pytest.raises(ValidationError):
        GeneralizedDiceScoreConfig(weight_type="abc")


def test_fails_validation_surface_dist():
    with pytest.raises(ValidationError):
        SurfaceDistanceMetricConfig(distance_metric="abc")


def test_fails_validation_haussdorf():
    with pytest.raises(ValidationError):
        HausdorffDistanceMetricConfig(percentile=-1)


def test_fails_validation_surface_dice():
    with pytest.raises(ValidationError):
        SurfaceDiceMetricConfig(class_thresholds=0.1)


def test_DiceMetricConfig():
    config = DiceMetricConfig(
        num_classes=3,
        include_background=False,
        reduction="mean",
    )
    assert config.name == "DiceMetric"
    assert config.num_classes == 3
    assert not config.include_background
    assert config.reduction == "mean"
    assert config.ignore_empty == "DefaultFromLibrary"
    assert not config.get_not_nans
    assert not config.return_with_label


def test_MeanIoUConfig():
    config = MeanIoUConfig(
        num_classes=3,
        include_background=False,
        reduction="mean",
    )
    assert config.name == "MeanIoU"
    assert not config.include_background
    assert config.reduction == "mean"
    assert config.ignore_empty == "DefaultFromLibrary"
    assert not config.get_not_nans


def test_GeneralizedDiceScoreConfig():
    config = GeneralizedDiceScoreConfig(
        weight_type="square",
        reduction="mean_batch",
    )
    assert config.name == "GeneralizedDiceScore"
    assert config.weight_type == "square"
    assert config.include_background == "DefaultFromLibrary"
    assert config.reduction == "mean_batch"


def test_SurfaceDistanceMetricConfig():
    config = SurfaceDistanceMetricConfig(
        symmetric=True,
        distance_metric="taxicab",
    )
    assert config.name == "SurfaceDistanceMetric"
    assert config.symmetric
    assert config.distance_metric == "taxicab"
    assert config.reduction == "DefaultFromLibrary"
    assert config.include_background == "DefaultFromLibrary"


def test_HausdorffDistanceMetricConfig():
    config = HausdorffDistanceMetricConfig(
        percentile=50,
        directed=True,
    )
    assert config.name == "HausdorffDistanceMetric"
    assert config.percentile == 50
    assert config.directed
    assert config.distance_metric == "DefaultFromLibrary"
    assert config.include_background == "DefaultFromLibrary"
    assert not config.get_not_nans


def test_SurfaceDiceMetricConfig():
    config = SurfaceDiceMetricConfig(
        use_subvoxels=True, class_thresholds=[0.1, 100], distance_metric="chessboard"
    )
    assert config.name == "SurfaceDiceMetric"
    assert config.class_thresholds == (0.1, 100)
    assert config.use_subvoxels
    assert config.distance_metric == "chessboard"
    assert config.include_background == "DefaultFromLibrary"
    assert not config.get_not_nans
