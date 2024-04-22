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
            "loss": "abc",
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "L1Loss",
            "selection_metrics": "loss",
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.reconstruction_config import ReconstructionConfig

    with pytest.raises(ValidationError):
        ReconstructionConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "L1Loss",
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "BCEWithLogitsLoss",
            "selection_metrics": ["loss"],
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.reconstruction_config import ReconstructionConfig

    ReconstructionConfig(**parameters)
