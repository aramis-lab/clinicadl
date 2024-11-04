import pytest
from pydantic import ValidationError

from clinicadl.metrics.config.reconstruction import (
    MultiScaleSSIMMetricConfig,
    PSNRMetricConfig,
    SSIMMetricConfig,
)


# PSNR #
@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"max_val": 255, "reduction": "abc"},
        {"max_val": 255, "get_not_nans": True},
        {"max_val": 0},
    ],
)
def test_fails_validation_psnr(bad_inputs):
    with pytest.raises(ValidationError):
        PSNRMetricConfig(**bad_inputs)


@pytest.mark.parametrize(
    "good_inputs",
    [
        {"max_val": 255, "reduction": "sum"},
        {"max_val": 255, "reduction": "mean"},
        {"max_val": 255, "get_not_nans": False},
    ],
)
def test_passes_validations_psnr(good_inputs):
    PSNRMetricConfig(**good_inputs)


def test_PSNRMetricConfig():
    config = PSNRMetricConfig(
        max_val=7,
        reduction="sum",
    )
    assert config.name == "PSNRMetric"
    assert config.max_val == 7
    assert config.reduction == "sum"
    assert not config.get_not_nans


# SSIM #
@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"spatial_dims": 1},
        {"spatial_dims": 2, "data_range": 0},
        {"spatial_dims": 2, "kernel_type": "abc"},
        {"spatial_dims": 2, "kernel_sigma": 0},
        {"spatial_dims": 2, "kernel_sigma": (1.0, 2.0, 3.0)},
        {"spatial_dims": 2, "k1": -1.0},
        {"spatial_dims": 2, "k2": -0.01},
    ],
)
def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        SSIMMetricConfig(**bad_inputs)
    with pytest.raises(ValidationError):
        MultiScaleSSIMMetricConfig(**bad_inputs)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"spatial_dims": 2, "win_size": 0},
        {"spatial_dims": 2, "win_size": (1, 2, 3)},
    ],
)
def test_fails_validations_ssim(bad_inputs):
    with pytest.raises(ValidationError):
        SSIMMetricConfig(**bad_inputs)


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"spatial_dims": 2, "kernel_size": 0},
        {"spatial_dims": 2, "kernel_size": (1, 2, 3)},
        {"spatial_dims": 2, "weights": (0.0, 1.0)},
        {"spatial_dims": 2, "weights": 1.0},
    ],
)
def test_fails_validation_msssim(bad_inputs):
    with pytest.raises(ValidationError):
        MultiScaleSSIMMetricConfig(**bad_inputs)


@pytest.mark.parametrize(
    "good_inputs",
    [
        {
            "spatial_dims": 2,
            "data_range": 1,
            "kernel_type": "gaussian",
            "win_size": 10,
            "kernel_sigma": 1.0,
            "k1": 1.0,
            "k2": 1.0,
            "weights": [1.0, 2.0],
        },
        {"spatial_dims": 2, "win_size": (1, 2), "kernel_sigma": (1.0, 2.0)},
    ],
)
def test_passes_validations(good_inputs):
    MultiScaleSSIMMetricConfig(**good_inputs)
    SSIMMetricConfig(**good_inputs)


def test_SSIMMetricConfig():
    config = SSIMMetricConfig(
        spatial_dims=2,
        reduction="sum",
        k1=1.0,
    )
    assert config.name == "SSIMMetric"
    assert config.reduction == "sum"
    assert config.spatial_dims == 2
    assert config.k1 == 1.0
    assert config.k2 == "DefaultFromLibrary"


def test_MultiScaleSSIMMetric():
    config = MultiScaleSSIMMetricConfig(
        spatial_dims=2, reduction="sum", k1=1.0, weights=[1.0], kernel_size=10
    )
    assert config.name == "MultiScaleSSIMMetric"
    assert config.reduction == "sum"
    assert config.spatial_dims == 2
    assert config.kernel_size == 10
    assert config.k1 == 1.0
    assert config.k2 == "DefaultFromLibrary"
    assert config.weights == (1.0,)
