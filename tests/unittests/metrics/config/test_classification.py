import pytest
from pydantic import ValidationError

from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)


# ROCAUC
def test_fails_validations_rocauc():
    with pytest.raises(ValidationError):
        ROCAUCMetricConfig(average="abc")


def test_ROCAUCMetricConfig():
    config = ROCAUCMetricConfig(
        average="macro",
    )
    assert config.name == "ROCAUCMetric"
    assert config.average == "macro"


# Confusion Matrix
@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"reduction": "abc"},
        {"get_not_nans": True},
    ],
)
def test_fails_validations_cmatrix(bad_inputs):
    with pytest.raises(ValidationError):
        ConfusionMatrixMetricConfig(**bad_inputs)


def test_ConfusionMatrixMetricConfig():
    config = ConfusionMatrixMetricConfig(
        metric_name="recall",
        reduction="sum",
    )
    assert config.name == "ConfusionMatrixMetric"
    assert config.reduction == "sum"
    assert config.metric_name == "recall"
    assert config.include_background == "DefaultFromLibrary"
    assert not config.get_not_nans
