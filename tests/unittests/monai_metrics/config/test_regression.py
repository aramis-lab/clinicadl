import pytest
from pydantic import ValidationError

from clinicadl.monai_metrics.config.regression import (
    MAEConfig,
    MSEConfig,
    RMSEConfig,
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
        MAEConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        MSEConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        RMSEConfig(**bad_inputs)


@pytest.mark.parametrize(
    "good_inputs",
    [
        {"reduction": "sum"},
        {"reduction": "mean"},
        {"get_not_nans": False},
    ],
)
def test_passes_validations(good_inputs):
    MAEConfig(**good_inputs)
    MSEConfig(**good_inputs)
    RMSEConfig(**good_inputs)


def test_MAEConfig():
    config = MAEConfig(
        reduction="sum",
    )
    assert config.metric == "MAEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans


def test_MSEConfig():
    config = MSEConfig(
        reduction="sum",
    )
    assert config.metric == "MSEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans


def test_RMSEConfig():
    config = RMSEConfig(
        reduction="sum",
    )
    assert config.metric == "RMSEMetric"
    assert config.reduction == "sum"
    assert not config.get_not_nans
