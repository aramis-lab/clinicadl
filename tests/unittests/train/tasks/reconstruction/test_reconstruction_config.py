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
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "selection_metrics": "loss",
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "normalization": "abc",
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.reconstruction import ReconstructionConfig

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
            "normalization": "batch",
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "selection_metrics": ["loss"],
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.reconstruction import ReconstructionConfig

    ReconstructionConfig(**parameters)
