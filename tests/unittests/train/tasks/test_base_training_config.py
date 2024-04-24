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
            "dropout": 1.1,
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "optimizer": "abc",
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "data_augmentation": ("abc",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": "AD",
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "size_reduction_factor": 1,
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
            "size_reduction_factor": 2,
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "diagnoses": ["AD", "CN"],
            "data_augmentation": False,
            "transfer_path": False,
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.base_training_config import BaseTaskConfig

    BaseTaskConfig(**parameters)
