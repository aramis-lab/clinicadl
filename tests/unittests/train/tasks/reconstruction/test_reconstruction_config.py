from pathlib import Path

import pytest
from pydantic import ValidationError

import clinicadl.config.config.train.reconstruction as reconstruction


# Tests for customed validators #
def test_validation_config():
    c = reconstruction.ValidationConfig(selection_metrics=["MAE"])
    assert c.selection_metrics == ("MAE",)


# Global tests on the TrainingConfig class #
@pytest.fixture
def caps_example():
    dir_ = Path(__file__).parents[3] / "ressources" / "caps_example"
    return dir_


@pytest.fixture
def dummy_arguments(caps_example):
    args = {
        "caps_directory": caps_example,
        "preprocessing_json": "preprocessing.json",
        "tsv_directory": "",
        "output_maps_directory": "",
    }
    return args


@pytest.fixture(
    params=[
        {"loss": "abc"},
        {"selection_metrics": ("abc",)},
        {"normalization": "abc"},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture
def good_inputs(dummy_arguments):
    options = {
        "loss": "HuberLoss",
        "selection_metrics": ("PSNR",),
        "normalization": "batch",
    }
    return {**dummy_arguments, **options}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        reconstruction.ReconstructionConfig(**bad_inputs)


def test_passes_validations(good_inputs):
    c = reconstruction.ReconstructionConfig(**good_inputs)
    assert c.model.loss == "HuberLoss"
    assert c.validation.selection_metrics == ("PSNR",)
    assert c.model.normalization == "batch"
    assert c.network_task == "reconstruction"
