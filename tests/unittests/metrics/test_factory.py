import pytest
from monai.metrics import SSIMMetric
from pydantic import ValidationError
from torch.nn import MSELoss

from clinicadl.metrics.config import ImplementedMetric, create_metric_config
from clinicadl.metrics.factory import (
    get_metric_config,
    get_metric_from_config,
)

MANDATORY_ARGS = {
    "spatial_dims": 3,
    "max_val": 1,
    "class_thresholds": [0.1, 0.2],
    "loss_fn": MSELoss(),
}


def test_get_metric_from_config():
    # test all metrics
    for metric in ImplementedMetric:
        config = create_metric_config(metric=metric)(**MANDATORY_ARGS)
        metric, _ = get_metric_from_config(config=config)

    # test arguments
    config = create_metric_config("SSIMMetric")(
        spatial_dims=3,
        data_range=1.0,
        kernel_type="gaussian",
        kernel_sigma=13.0,
    )
    metric, updated_config = get_metric_from_config(config=config)
    assert isinstance(metric, SSIMMetric)
    assert metric.spatial_dims == 3
    assert metric.data_range == 1.0
    assert metric.kernel_type == "gaussian"
    assert metric.kernel_sigma == (13.0, 13.0, 13.0)

    assert updated_config.name == "SSIMMetric"
    assert updated_config.spatial_dims == 3
    assert updated_config.data_range == 1.0
    assert updated_config.kernel_type == "gaussian"
    assert updated_config.kernel_sigma == 13.0
    assert updated_config.k1 == 0.01


def test_get_metric_config():
    config = get_metric_config("SSIMMetric", spatial_dims=3)
    assert config.name == "SSIMMetric"
    assert config.kernel_sigma == 1.5
    assert config.spatial_dims == 3

    with pytest.raises(ValueError):
        get_metric_config("abc", **MANDATORY_ARGS)
