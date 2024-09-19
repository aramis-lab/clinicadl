import pytest
from pydantic import ValidationError

from clinicadl.monai_metrics.config.generation import MMDMetricConfig


def test_fails_validation():
    with pytest.raises(ValidationError):
        MMDMetricConfig(kernel_bandwidth=0)


def test_MMDMetricConfig():
    config = MMDMetricConfig(
        kernel_bandwidth=2.0,
    )
    assert config.metric == "MMDMetric"
    assert config.kernel_bandwidth == 2.0
