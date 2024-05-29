from pathlib import Path

import pytest
from pydantic import ValidationError


# Test RandomSearchConfig #
def test_random_search_config():
    from clinicadl.random_search.random_search_config import RandomSearchConfig

    config = RandomSearchConfig(
        first_conv_width=[1, 2],
        n_convblocks=1,
        n_fcblocks=(1,),
    )
    assert config.first_conv_width == (1, 2)
    assert config.n_convblocks == (1,)
    assert config.n_fcblocks == (1,)
    with pytest.raises(ValidationError):
        config.first_conv_width = (1, 0)


# Test Training Configs #
@pytest.fixture
def caps_example():
    dir_ = Path(__file__).parents[1] / "ressources" / "caps_example"
    return dir_


@pytest.fixture
def dummy_arguments(caps_example):
    args = {
        "caps_directory": caps_example,
        "preprocessing_json": "preprocessing.json",
        "tsv_directory": "",
        "maps_dir": "",
    }
    return args


@pytest.fixture
def random_model_arguments():
    args = {
        "convolutions_dict": {
            "conv0": {
                "in_channels": 1,
                "out_channels": 8,
                "n_conv": 2,
                "d_reduction": "MaxPooling",
            },
            "conv1": {
                "in_channels": 8,
                "out_channels": 16,
                "n_conv": 3,
                "d_reduction": "MaxPooling",
            },
        },
        "n_fcblocks": 2,
    }
    return args


def test_training_config(dummy_arguments, random_model_arguments):
    from clinicadl.random_search.random_search_config import ClassificationConfig

    config = ClassificationConfig(**dummy_arguments, **random_model_arguments)
    assert config.model.convolutions_dict == random_model_arguments["convolutions_dict"]
    assert config.model.n_fcblocks == random_model_arguments["n_fcblocks"]
    assert config.model.architecture == "RandomArchitecture"
    assert config.network_task == "classification"
    with pytest.raises(ValidationError):
        config.model.architecture = "abc"
