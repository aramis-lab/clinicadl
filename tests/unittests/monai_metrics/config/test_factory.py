import pytest

from clinicadl.monai_metrics.config import ImplementedMetrics, create_metric_config


def test_create_training_config():
    for metric in [e.value for e in ImplementedMetrics]:
        if metric == "Loss":
            with pytest.raises(ValueError):
                create_metric_config(metric)
        else:
            create_metric_config(metric)

    config_class = create_metric_config("Hausdorff distance")
    config = config_class(
        include_background=True,
        distance_metric="taxicab",
        reduction="sum",
        percentile=50,
    )
    assert config.metric == "HausdorffDistanceMetric"
    assert config.include_background
    assert config.distance_metric == "taxicab"
    assert config.reduction == "sum"
    assert config.percentile == 50
    assert config.directed == "DefaultFromLibrary"
    assert not config.get_not_nans

    config_class = create_metric_config("F1 score")
    config = config_class(
        include_background=True,
        compute_sample=True,
    )
    assert config.metric == "ConfusionMatrixMetric"
    assert config.include_background
    assert config.compute_sample
    assert config.metric_name == "f1 score"
