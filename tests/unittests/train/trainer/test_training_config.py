from pathlib import Path

import pytest
from pydantic import ValidationError

import clinicadl.train.trainer.training_config as config


# Tests for customed validators #
@pytest.fixture
def caps_example():
    dir_ = Path(__file__).parents[2] / "ressources" / "caps_example"
    return dir_


def test_cross_validation_config():
    c = config.CrossValidationConfig(
        split=[0],
        tsv_directory="",
    )
    assert c.split == (0,)


def test_data_config(caps_example):
    c = config.DataConfig(
        caps_directory=caps_example,
        preprocessing_json="preprocessing.json",
        diagnoses=["AD"],
    )
    expected_preprocessing_dict = {
        "preprocessing": "t1-linear",
        "mode": "image",
        "use_uncropped_image": False,
        "prepare_dl": False,
        "extract_json": "t1-linear_mode-image.json",
        "file_type": {
            "pattern": "*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "description": "T1W Image registered using t1-linear and cropped (matrix size 169\u00d7208\u00d7179, 1 mm isotropic voxels)",
            "needed_pipeline": "t1-linear",
        },
    }
    assert c.diagnoses == ("AD",)
    assert (
        c.preprocessing_dict == expected_preprocessing_dict
    )  # TODO : add test for multi-cohort
    assert c.mode == "image"
    with pytest.raises(ValidationError):
        c.preprocessing_dict = {"abc": "abc"}
    with pytest.raises(FileNotFoundError):
        c.preprocessing_json = ""
    c.preprocessing_json = None
    # c.preprocessing_dict = {"abc": "abc"}
    # assert c.preprocessing_dict == {"abc": "abc"}


def test_model_config():
    with pytest.raises(ValidationError):
        config.ModelConfig(
            **{
                "architecture": "",
                "loss": "",
                "dropout": 1.1,
            }
        )


def test_ssda_config(caps_example):
    preprocessing_json_target = (
        caps_example / "tensor_extraction" / "preprocessing.json"
    )
    c = config.SSDAConfig(
        ssda_network=True,
        preprocessing_json_target=preprocessing_json_target,
    )
    expected_preprocessing_dict = {
        "preprocessing": "t1-linear",
        "mode": "image",
        "use_uncropped_image": False,
        "prepare_dl": False,
        "extract_json": "t1-linear_mode-image.json",
        "file_type": {
            "pattern": "*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "description": "T1W Image registered using t1-linear and cropped (matrix size 169\u00d7208\u00d7179, 1 mm isotropic voxels)",
            "needed_pipeline": "t1-linear",
        },
    }
    assert c.preprocessing_dict_target == expected_preprocessing_dict
    c = config.SSDAConfig()
    assert c.preprocessing_dict_target == {}


def test_transferlearning_config():
    c = config.TransferLearningConfig(transfer_path=False)
    assert c.transfer_path is None


def test_transforms_config():
    c = config.TransformsConfig(data_augmentation=False)
    assert c.data_augmentation == ()
    c = config.TransformsConfig(data_augmentation=["Noise"])
    assert c.data_augmentation == ("Noise",)


# Global tests on the TrainingConfig class #
@pytest.fixture
def dummy_arguments(caps_example):
    args = {
        "caps_directory": caps_example,
        "preprocessing_json": "preprocessing.json",
        "tsv_directory": "",
        "maps_dir": "",
        "architecture": "",
        "loss": "",
        "selection_metrics": (),
    }
    return args


@pytest.fixture
def training_config():
    from pydantic import computed_field

    class TrainingConfig(config.TrainingConfig):
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
    assert c.cross_validation.n_splits == 7
    assert c.optimizer.optimizer == "Adagrad"
    assert c.transforms.data_augmentation == ("Smoothing",)
    assert c.data.diagnoses == ("AD",)
    assert c.dataloader.batch_size == 1
    assert c.transforms.size_reduction_factor == 5
    assert c.optimizer.learning_rate == 1e-1
    assert c.cross_validation.split == (0,)
    assert c.early_stopping.tolerance == 0.0


# Test config manipulation #
def test_assignment(dummy_arguments, training_config):
    c = training_config(**dummy_arguments)
    c.computational = {"gpu": False}
    c.dataloader = config.DataLoaderConfig(**{"batch_size": 1})
    c.dataloader.n_proc = 10
    with pytest.raises(ValidationError):
        c.computational = config.DataLoaderConfig()
    with pytest.raises(ValidationError):
        c.dataloader = {"sampler": "abc"}
    assert not c.computational.gpu
    assert c.dataloader.batch_size == 1
    assert c.dataloader.n_proc == 10
