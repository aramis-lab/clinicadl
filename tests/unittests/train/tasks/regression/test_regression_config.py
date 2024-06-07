from pathlib import Path

import pytest
from pydantic import ValidationError

import clinicadl.trainer.config.regression as regression


# Tests for customed validators #
def test_validation_config():
    c = regression.ValidationConfig(selection_metrics=["R2_score"])
    assert c.selection_metrics == ("R2_score",)


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
        "maps_dir": "",
        "gpu": False,
    }
    return args


@pytest.fixture(
    params=[
        {"loss": "abc"},
        {"selection_metrics": ("abc",)},
        {"selection_metrics": "R2_score"},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture
def good_inputs(dummy_arguments):
    options = {
        "loss": "KLDivLoss",
        "selection_metrics": ("R2_score",),
    }
    return {**dummy_arguments, **options}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        regression.RegressionConfig(**bad_inputs)


def test_passes_validations(good_inputs):
    c = regression.RegressionConfig(**good_inputs)
    assert c.model.loss == "KLDivLoss"
    assert c.validation.selection_metrics == ("R2_score",)
    assert c.network_task == "regression"


def test_update_from_toml(dummy_arguments):
    toml_path = (
        Path(__file__).parents[3] / "ressources" / "functional_config_example.toml"
    )
    c = regression.RegressionConfig(**dummy_arguments)
    c.update_with_toml(toml_path)
    assert not c.computational.gpu
    assert c.model.loss == "SmoothL1Loss"
