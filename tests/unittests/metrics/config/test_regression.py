import pytest
from pydantic import ValidationError

from clinicadl.metrics.config.regression import (
    MAEMetricConfig,
    MSEMetricConfig,
    RMSEMetricConfig,
)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"reduction": "abc"},
        {"get_not_nans": True},
    ],
)
def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        MAEMetricConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        MSEMetricConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        RMSEMetricConfig(**bad_inputs)


@pytest.mark.parametrize(
    "good_inputs",
    [
        {"reduction": "sum"},
        {"reduction": "mean"},
        {"get_not_nans": False},
    ],
)
def test_passes_validations(good_inputs):
    MAEMetricConfig(**good_inputs)
    MSEMetricConfig(**good_inputs)
    RMSEMetricConfig(**good_inputs)


def test_MAEMetricConfig():
    config = MAEMetricConfig(
        reduction="sum",
    )
    assert config.name == "MAEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans


def test_MSEMetricConfig():
    config = MSEMetricConfig(
        reduction="sum",
    )
    assert config.name == "MSEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans


def test_RMSEMetricConfig():
    config = RMSEMetricConfig(
        reduction="sum",
    )
    assert config.name == "RMSEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans
