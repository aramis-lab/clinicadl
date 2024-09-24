import pytest
from pydantic import ValidationError

from clinicadl.monai_metrics.config.classification import (
    ROCAUCConfig,
    create_confusion_matrix_config,
)
from clinicadl.monai_metrics.config.enum import ConfusionMatrixMetric


# ROCAUC
def test_fails_validations_rocauc():
    with pytest.raises(ValidationError):
        ROCAUCConfig(average="abc")


def test_ROCAUCConfig():
    config = ROCAUCConfig(
        average="macro",
    )
    assert config.metric == "ROCAUCMetric"
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
    for m in ConfusionMatrixMetric:
        config_class = create_confusion_matrix_config(m.value)
        with pytest.raises(ValidationError):
            config_class(**bad_inputs)


def test_passes_validations_cmatrix():
    for m in ConfusionMatrixMetric:
        config_class = create_confusion_matrix_config(m.value)
        config_class(
            reduction="mean",
            get_not_nans=False,
            compute_sample=False,
        )


def test_ConfusionMatrixMetricConfig():
    for m in ConfusionMatrixMetric:
        config_class = create_confusion_matrix_config(m.value)
        config = config_class(
            reduction="sum",
        )
        assert config.metric == "ConfusionMatrixMetric"
        assert config.reduction == "sum"
        assert config.metric_name == m.value
        assert config.include_background == "DefaultFromLibrary"
        assert not config.get_not_nans
