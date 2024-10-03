from pathlib import Path

import pytest
from pydantic import ValidationError

import clinicadl.trainer.config.classification as classification


# Tests for customed class methods #
def test_model_config():
    with pytest.raises(ValidationError):
        classification.NetworkConfig(
            **{
                "architecture": "",
                "loss": "",
                "selection_threshold": 1.1,
            }
        )


def test_validation_config():
    c = classification.ValidationConfig(selection_metrics=["accuracy"])
    assert c.selection_metrics == ("accuracy",)


# Global tests #
@pytest.fixture
def caps_example():
    dir_ = Path(__file__).parents[3] / "ressources" / "caps_example"
    return dir_


@pytest.fixture
def dummy_arguments(caps_example):
    args = {
        "caps_directory": caps_example,
        "preprocessing_json": "preprocessing.json",
        "tsv_path": "",
        "maps_dir": "",
        "gpu": False,
    }
    return args


@pytest.fixture(
    params=[
        {"loss": "abc"},
        {"selection_metrics": ("abc",)},
        {"selection_metrics": "F1_score"},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture
def good_inputs(dummy_arguments):
    options = {
        "loss": "MultiMarginLoss",
        "selection_metrics": ("F1_score",),
        "selection_threshold": 0.5,
    }
    return {**dummy_arguments, **options}


def test_fails_validations(bad_inputs):
    with pytest.raises(ValidationError):
        classification.ClassificationConfig(**bad_inputs)


def test_passes_validations(good_inputs):
    c = classification.ClassificationConfig(**good_inputs)
    assert c.model.loss == "MultiMarginLoss"
    assert c.validation.selection_metrics == ("F1_score",)
    assert c.model.selection_threshold == 0.5
    assert c.network_task == "classification"


def test_update_from_toml(dummy_arguments):
    toml_path = (
        Path(__file__).parents[3] / "ressources" / "functional_config_example.toml"
    )
    c = classification.ClassificationConfig(**dummy_arguments)
    c.update_with_toml(toml_path)
    assert not c.computational.gpu
    assert c.model.loss == "MultiMarginLoss"
