import pytest

from clinicadl.metrics.config import *
from clinicadl.metrics.factory import get_metric_from_dict

MANDATORY_ARGS = {
    "max_val": 1,
    "class_thresholds": (0.5, 0.5),
    "spatial_dims": 2,
}


@pytest.mark.parametrize(
    "config",
    [
        ConfusionMatrixMetricConfig,
        ROCAUCMetricConfig,
        AveragePrecisionMetricConfig,
        MultiScaleSSIMMetricConfig,
        PSNRMetricConfig,
        SSIMMetricConfig,
        MAEMetricConfig,
        MSEMetricConfig,
        RMSEMetricConfig,
        DiceMetricConfig,
        GeneralizedDiceScoreConfig,
        HausdorffDistanceMetricConfig,
        MeanIoUConfig,
        SurfaceDiceMetricConfig,
        SurfaceDistanceMetricConfig,
        LossMetricConfig,
    ],
)
def test_get_metric_from_dict(config):
    c = config(**MANDATORY_ARGS)
    config_dict = c.to_dict()
    c = get_metric_from_dict(config_dict)
    assert isinstance(c, config)

    if config is PSNRMetricConfig:
        c = PSNRMetricConfig(max_val=1)
        assert get_metric_from_dict(c.to_dict()).max_val == 1.0
