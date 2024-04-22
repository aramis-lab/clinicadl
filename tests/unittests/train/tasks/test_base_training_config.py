import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ("AD", "CN"),
            "optimizer": "Adam",
            "dropout": 1.1,
            "data_augmentation": ("Noise",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ("AD", "CN"),
            "optimizer": "abc",
            "dropout": 0.5,
            "data_augmentation": ("Noise",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ("AD", "CN"),
            "optimizer": "Adam",
            "dropout": 0.5,
            "data_augmentation": ("abc",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": "AD",
            "optimizer": "Adam",
            "dropout": 0.5,
            "data_augmentation": ("Noise",),
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.base_training_config import BaseTaskConfig

    with pytest.raises(ValidationError):
        BaseTaskConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ("AD", "CN"),
            "optimizer": "Adam",
            "dropout": 0.5,
            "data_augmentation": ("Noise",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ["AD", "CN"],
            "optimizer": "Adam",
            "dropout": 0.5,
            "data_augmentation": ("Noise",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ["AD", "CN"],
            "optimizer": "Adam",
            "dropout": 0.5,
            "data_augmentation": ("Noise",),
            "transfer_path": False,
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.base_training_config import BaseTaskConfig

    BaseTaskConfig(**parameters)
