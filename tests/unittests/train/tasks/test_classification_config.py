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
            "loss": "CrossEntropyLoss",
            "selection_threshold": 1.1,
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "abc",
            "selection_threshold": 0.5,
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "CrossEntropyLoss",
            "selection_threshold": 0.5,
            "selection_metrics": "loss",
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.classification_config import ClassificationConfig

    with pytest.raises(ValidationError):
        ClassificationConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "CrossEntropyLoss",
            "selection_threshold": 0.5,
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "MultiMarginLoss",
            "selection_threshold": 0.5,
            "selection_metrics": ["loss"],
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.classification_config import ClassificationConfig

    ClassificationConfig(**parameters)
