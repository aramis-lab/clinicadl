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
            "selection_metrics": ["abc"],
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.regression import RegressionConfig

    with pytest.raises(ValidationError):
        RegressionConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "MSELoss",
            "selection_metrics": ("loss",),
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
    from clinicadl.train.tasks.regression import RegressionConfig

    RegressionConfig(**parameters)
