import pytest
from pydantic import ValidationError

from clinicadl.monai_metrics.config.segmentation import (
    DiceConfig,
    GeneralizedDiceConfig,
    HausdorffDistanceConfig,
    IoUConfig,
    SurfaceDiceConfig,
    SurfaceDistanceConfig,
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
        DiceConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        IoUConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        SurfaceDistanceConfig(**bad_inputs)


def test_fails_validation_dice():
    with pytest.raises(ValidationError):
        DiceConfig(return_with_label=True)
    with pytest.raises(ValidationError):
        DiceConfig(num_classes=0)


def test_fails_validation_gen_dice():
    with pytest.raises(ValidationError):
        GeneralizedDiceConfig(reduction="mean")
    with pytest.raises(ValidationError):
        GeneralizedDiceConfig(weight_type="abc")


def test_fails_validation_surface_dist():
    with pytest.raises(ValidationError):
        SurfaceDistanceConfig(distance_metric="abc")


def test_fails_validation_haussdorf():
    with pytest.raises(ValidationError):
        HausdorffDistanceConfig(percentile=-1)


def test_fails_validation_surface_dice():
    with pytest.raises(ValidationError):
        SurfaceDiceConfig(class_thresholds=0.1)


def test_DiceConfig():
    config = DiceConfig(
        num_classes=3,
        include_background=False,
        reduction="mean",
    )
    assert config.metric == "DiceMetric"
    assert config.num_classes == 3
    assert not config.include_background
    assert config.reduction == "mean"
    assert config.ignore_empty == "DefaultFromLibrary"
    assert not config.get_not_nans
    assert not config.return_with_label


def test_IoUConfig():
    config = IoUConfig(
        num_classes=3,
        include_background=False,
        reduction="mean",
    )
    assert config.metric == "MeanIoU"
    assert not config.include_background
    assert config.reduction == "mean"
    assert config.ignore_empty == "DefaultFromLibrary"
    assert not config.get_not_nans


def test_GeneralizedDiceConfig():
    config = GeneralizedDiceConfig(
        weight_type="square",
        reduction="mean_batch",
    )
    assert config.metric == "GeneralizedDiceScore"
    assert config.weight_type == "square"
    assert config.include_background == "DefaultFromLibrary"
    assert config.reduction == "mean_batch"


def test_SurfaceDistanceConfig():
    config = SurfaceDistanceConfig(
        symmetric=True,
        distance_metric="taxicab",
    )
    assert config.metric == "SurfaceDistanceMetric"
    assert config.symmetric
    assert config.distance_metric == "taxicab"
    assert config.reduction == "DefaultFromLibrary"
    assert config.include_background == "DefaultFromLibrary"


def test_HausdorffDistanceConfig():
    config = HausdorffDistanceConfig(
        percentile=50,
        directed=True,
    )
    assert config.metric == "HausdorffDistanceMetric"
    assert config.percentile == 50
    assert config.directed
    assert config.distance_metric == "DefaultFromLibrary"
    assert config.include_background == "DefaultFromLibrary"
    assert not config.get_not_nans


def test_SurfaceDiceConfig():
    config = SurfaceDiceConfig(
        use_subvoxels=True, class_thresholds=[0.1, 100], distance_metric="chessboard"
    )
    assert config.metric == "SurfaceDiceMetric"
    assert config.class_thresholds == (0.1, 100)
    assert config.use_subvoxels
    assert config.distance_metric == "chessboard"
    assert config.include_background == "DefaultFromLibrary"
    assert not config.get_not_nans
