import pytest

from clinicadl.metrics.config import ImplementedMetric, create_metric_config


def test_create_training_config():
    for metric in [e.value for e in ImplementedMetric]:
        if metric == "Loss":
            with pytest.raises(ValueError):
                create_metric_config(metric)
        else:
            create_metric_config(metric)

    config_class = create_metric_config("HausdorffDistanceMetric")
    config = config_class(
        include_background=True,
        distance_metric="taxicab",
        reduction="sum",
        percentile=50,
    )
    assert config.name == "HausdorffDistanceMetric"
    assert config.include_background
    assert config.distance_metric == "taxicab"
    assert config.reduction == "sum"
    assert config.percentile == 50
    assert config.directed == "DefaultFromLibrary"
    assert not config.get_not_nans
