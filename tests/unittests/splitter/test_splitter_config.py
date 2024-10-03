from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.config.config.ssda import SSDAConfig
from clinicadl.network.config import NetworkConfig
from clinicadl.splitter.config import SplitConfig, SplitterConfig
from clinicadl.splitter.validation import ValidationConfig
from clinicadl.trainer.transfer_learning import TransferLearningConfig
from clinicadl.transforms.config import TransformsConfig


# Tests for customed validators #
@pytest.fixture
def caps_example():
    dir_ = Path(__file__).parents[2] / "ressources" / "caps_example"
    return dir_


def test_split_config():
    c = SplitConfig(
        n_splits=3,
        split=[0],
        tsv_path="",
    )
    assert c.split == (0,)


def test_validation_config():
    c = ValidationConfig(
        evaluation_steps=3,
        valid_longitudinal=True,
    )
    assert not c.skip_leak_check
    assert c.selection_metrics == ()


# Global tests on the TrainingConfig class #
@pytest.fixture
def dummy_arguments(caps_example):
    args = {
        "caps_directory": caps_example,
        "preprocessing_json": "preprocessing.json",
        "tsv_path": "",
        "maps_dir": "",
        "gpu": False,
        "architecture": "",
        "loss": "",
        "selection_metrics": (),
    }
    return args


@pytest.fixture
def splitter_config():
    from pydantic import computed_field

    from clinicadl.splitter.config import SplitterConfig

    class TrainingConfig(TrainConfig):
        @computed_field
        @property
        def network_task(self) -> str:
            return ""

    return TrainingConfig


@pytest.fixture(
    params=[
        {"gpu": "abc"},
        {"n_splits": -1},
        {"optimizer": "abc"},
        {"data_augmentation": ("abc",)},
        {"diagnoses": "AD"},
        {"batch_size": 0},
        {"size_reduction_factor": 1},
        {"learning_rate": 0.0},
        {"split": [-1]},
        {"tolerance": -0.01},
    ]
)
def bad_inputs(request, dummy_arguments):
    return {**dummy_arguments, **request.param}


@pytest.fixture
def good_inputs(dummy_arguments):
    options = {
        "gpu": False,
        "n_splits": 7,
        "optimizer": "Adagrad",
        "data_augmentation": ("Smoothing",),
        "diagnoses": ("AD",),
        "batch_size": 1,
        "size_reduction_factor": 5,
        "learning_rate": 1e-1,
        "split": [0],
        "tolerance": 0.0,
    }
    return {**dummy_arguments, **options}


def test_fails_validations(bad_inputs, training_config):
    with pytest.raises(ValidationError):
        training_config(**bad_inputs)


def test_passes_validations(good_inputs, training_config):
    c = training_config(**good_inputs)
    assert not c.computational.gpu
    assert c.split.n_splits == 7
    assert c.optimizer.optimizer == "Adagrad"
    assert c.transforms.data_augmentation == ("Smoothing",)
    assert c.data.diagnoses == ("AD",)
    assert c.dataloader.batch_size == 1
    assert c.transforms.size_reduction_factor == 5
    assert c.optimizer.learning_rate == 1e-1
    assert c.split.split == (0,)
    assert c.early_stopping.tolerance == 0.0


# Test config manipulation #
def test_assignment(dummy_arguments, training_config):
    c = training_config(**dummy_arguments)
    c.computational = {"gpu": False}
    c.dataloader = DataLoaderConfig(**{"batch_size": 1})
    c.dataloader.n_proc = 10
    with pytest.raises(ValidationError):
        c.computational = DataLoaderConfig()
    with pytest.raises(ValidationError):
        c.dataloader = {"sampler": "abc"}
    assert not c.computational.gpu
    assert c.dataloader.batch_size == 1
    assert c.dataloader.n_proc == 10
